# Copyright 2025 Chinese Information Processing Laboratory, Institute of Software, Chinese Academy of Sciences.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
from typing import Any, Dict, List
import re
import os
from datetime import datetime
import tempfile
import uuid
import base64

import structlog
from fastapi import HTTPException

from sandbox.configs.run_config import RunConfig
from sandbox.datasets.types import EvalTestCase, GeneralStdioTest, RunStatus, TestConfig
from sandbox.runners.types import compile_languages
from sandbox.utils.common import truncate_str
from sandbox.utils.execution import max_concurrency
from sandbox.utils.sandbox_client import RunCodeRequest, run_code_in_sandbox, run_code_in_sandbox_w_retry
from sandbox.server.sandbox_api import RunCodeResponse

sandbox_config = RunConfig.get_instance_sync()
logger = structlog.stdlib.get_logger()


async def check_auto_test_case(code: str, config: TestConfig) -> EvalTestCase:
    '''
    auto test: run the code and check if the return value is 0
    '''
    result = await run_code_in_sandbox(RunCodeRequest(code=code, language=config.language))
    return EvalTestCase(passed=result.status == RunStatus.Success, exec_info=result)


async def check_function_call_test_case(code: str, case: GeneralStdioTest, config: TestConfig) -> EvalTestCase:
    '''
    auto test: run the code and check if the return value is 0
    '''
    result = await run_code_in_sandbox(RunCodeRequest(code=code, 
                                                      language=config.language,
                                                      compile_timeout=config.compile_timeout or 10,
                                                      run_timeout=config.run_timeout or 10,
                                                      mem_limit=config.mem_limit,))
    return EvalTestCase(passed=result.status == RunStatus.Success, exec_info=result, test_info=case)

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def float_equal(a, b, rel_tol=1e-5):
    return abs(a - b) / max(abs(b), 1e-10) < rel_tol

async def check_special_judge(stdin: str, stdout: str, answer: str, special_judge_program: str, special_judge_language: str = "python") -> bool:
    stdin_base64 = base64.b64encode(stdin.encode('utf-8')).decode('utf-8')
    stdout_base64 = base64.b64encode(stdout.encode('utf-8')).decode('utf-8')
    answer_base64 = base64.b64encode(answer.encode('utf-8')).decode('utf-8')
    files = {
        "stdin.txt": stdin_base64,
        "stdout.txt": stdout_base64,
        "answer.txt": answer_base64,
    }
    
    result = await run_code_in_sandbox_w_retry(
        RunCodeRequest(code=special_judge_program,
                       language=special_judge_language,
                       files=files,
                       stdin=None,
                       compile_timeout=10,
                       run_timeout=10,))
    logger.debug(f"check_special_judge result: {result}")

    return result.status == RunStatus.Success and result.run_result.return_code == 0


async def check_stdio_test_case(code: str, case: GeneralStdioTest, config: TestConfig, lower_cmp=True) -> EvalTestCase:
    if config.language in compile_languages:
        result = await run_code_in_sandbox_w_retry(
            RunCodeRequest(code=code,
                           language=config.language,
                           stdin=case.input['stdin'],
                           compile_timeout=config.compile_timeout or 10,
                           run_timeout=config.run_timeout or 10,
                           mem_limit=config.mem_limit,))
    else:
        result = await run_code_in_sandbox_w_retry(
            RunCodeRequest(code=code,
                           language=config.language,
                           stdin=case.input['stdin'],
                           run_timeout=config.run_timeout or 20,
                           mem_limit=config.mem_limit,))
    fail_case = EvalTestCase(passed=False, exec_info=result, test_info=case.model_dump())
    if result.status != 'Success':
        return fail_case
    result_lines = result.run_result.stdout.strip().split('\n')
    expected_lines = case.output['stdout'].strip().split('\n')
    if len(result_lines) - len(expected_lines) == 1 and result_lines[-1] == '':
        result_lines = result_lines[:-1]
    if len(expected_lines) - len(result_lines) == 1 and expected_lines[-1] == '':
        expected_lines = expected_lines[:-1]
    # if len(result_lines) != len(expected_lines):
    #     return fail_case
    special_judge_program = config.extra.get('special_judge_program', None)
    special_judge_language = config.extra.get('special_judge_language', 'python')
    for rl, el in zip(result_lines, expected_lines):
        if lower_cmp:
            rl = rl.lower()
            el = el.lower()
        if rl.strip() != el.strip():
            if is_float(el) and is_float(rl):
                if float_equal(float(rl), float(el)):
                    continue
            if special_judge_program is not None:
                correct = await check_special_judge(
                    stdin=case.input['stdin'],
                    stdout=case.output['stdout'],
                    answer=result.run_result.stdout,
                    special_judge_program=special_judge_program,
                    special_judge_language=special_judge_language
                )
                if correct:
                    return EvalTestCase(passed=True, exec_info=result, test_info=case.model_dump())
                else:
                    return fail_case
            else:
                return fail_case
    if not config.extra.get('return_full_case', False):
        for k in case.input:
            case.input[k] = truncate_str(case.input[k])
        for k in case.output:
            case.output[k] = truncate_str(case.output[k])
    return EvalTestCase(passed=True, exec_info=result, test_info=case.model_dump())


async def check_stdio_test_cases(code: str,
                                 cases: List[GeneralStdioTest],
                                 config: TestConfig,
                                 lower_cmp=True) -> List[EvalTestCase]:
    result = []
    for case in cases:
        outcome = await check_stdio_test_case(code, case, config, lower_cmp)
        result.append(outcome)
        if not outcome.passed:
            break
    return result


async def check_stdio_test_cases_parallel(code: str,
                                          cases: List[GeneralStdioTest],
                                          config: TestConfig,
                                          lower_cmp=True) -> List[EvalTestCase]:
    instance_id = uuid.uuid4().hex
    instance_logger = logger.bind(instance_id=instance_id)
    result = []
    tasks: List[asyncio.Task[EvalTestCase]] = []

    check_stdio_test_case_limited = check_stdio_test_case
    if sandbox_config.dataset.max_runner_concurrency > 0:
        check_stdio_test_case_limited = max_concurrency(
            sandbox_config.dataset.max_runner_concurrency)(check_stdio_test_case)

    for case in cases:
        task = asyncio.create_task(check_stdio_test_case_limited(code, case, config, lower_cmp))
        tasks.append(task)

    run_all_cases = config.extra.get("run_all_cases", False)
    total_timeout = config.extra.get("total_timeout", 300)
    deadline = asyncio.get_running_loop().time() + total_timeout

    for task_idx, task in enumerate(tasks):
        instance_logger.info(f"check_stdio_test_cases_parallel, total_timeout: {total_timeout}, task_idx: {task_idx}, deadline: {deadline}, current_time: {asyncio.get_running_loop().time()}")
        if asyncio.get_running_loop().time() > deadline:
            instance_logger.error("check_stdio_test_cases_parallel timeout", total_timeout=total_timeout, task_idx=task_idx)
            for remaining_task in tasks:
                if not remaining_task.done():
                    time_out_outcome = EvalTestCase(passed=False, exec_info=RunCodeResponse(status=RunStatus.SandboxError, message= "Total Timeout"), test_info=None)
                    result.append(time_out_outcome)
                    remaining_task.cancel()
            break
        try:
            outcome = await task
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Failed to check stdio call test case: {e}')
        result.append(outcome)

        if not run_all_cases and not outcome.passed:
            for remaining_task in tasks:
                if not remaining_task.done():
                    remaining_task.cancel()
            break

    return result

def concat_function_assertion(function, fn_name, input, output, language):
    original_output = output
    if isinstance(output, list) and len(output) == 1:
        output = output[0]
    if type(output) is str:
        output = f'"{output}"'
    elif type(output) is bool:
        if language in ["cpp", "java", "csharp", "go", "rust", "D_ut", "lua", "julia", "nodejs", "typescript", "php", "ruby", "scala", "swift"]:
            output = "true" if output else "false"
        elif language == "racket":
            output = "#t" if output else "#f"
        elif language in ["bash", "perl"]:
            output = "1" if output else "0"
    input_str = ', '.join([f'"{i}"' if type(i) is str else str(i) for i in input])
    # python, racket, D_ut, lua, julia, nodejs, cpp, go, java, typescript, csharp, rust, php, bash, ruby, perl, scala, swift
    if language == "python":
        if "class Solution" not in function:
            full_code = f'''
{function}
res = {fn_name}(*{input})
assert res == {original_output}[0]
'''
        else:
            full_code = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n"
            full_code += f'''
{function}
sol = Solution()
res = sol.{fn_name}(*{input})
assert res == {original_output}[0]
'''
    elif language == "racket":
        input_str_racket = ' '.join([f'"{i}"' if type(i) is str else str(i) for i in input])
        full_code = f'''
{function}
(define (main)
    (define res ({fn_name} {input_str_racket}))
    (define expected {output})
    (if (equal? res expected)
        (display "Test passed")
        (error "Test failed")))
(main)
'''
    elif language == "D_ut":
        full_code = f'''
import std.stdio;
{function}
void main() {{
    auto res = {fn_name}({input_str});
    auto expected = {output};
    if (res != expected) {{
        throw new Exception("Test failed");
    }} else {{
        writeln("Test passed");
    }}
}}
'''
    elif language == "lua":
        full_code = f'''
{function}
res = {fn_name}({input_str})
assert(res =={output}, "Test failed: expected 8, got " .. res)
'''
    elif language == "julia":
        full_code = f'''
{function}
res = {fn_name}({input_str})
if res != {output}
    error("Test failed")
end
'''
    elif language == "nodejs":
        full_code = f'''
{function}
const res = {fn_name}({input_str});
const expected = {output};
if (res !== expected) {{
    throw new Error(`Test failed`);
}}
'''
    elif language == "cpp":
        full_code = f'''
#include <iostream>
#include <cassert>
{function}
int main() {{
    auto res = {fn_name}({input_str});
    auto expected = {output};
    assert(res == expected);
    return 0;
}}
'''
    elif language == "go":
        full_code = f'''
package main
import (
    "fmt"
    "testing"
)
{function}
func Test{fn_name}(t *testing.T) {{
    res := {fn_name}({input_str})
    expected := {output}
    if res != expected {{
        t.Errorf("Expected %v, got %v", expected, res)
    }}
}}
func main() {{
    res := {fn_name}({input_str})
    expected := {output}
    if res != expected {{
        panic(fmt.Sprintf("Test failed: Expected %v, got %v", expected, res))
    }} else {{
        fmt.Println("Test passed")
    }}
}}
'''
    elif language == "java":
        full_code = f'''
public class Main {{
    {function}
    public static void main(String[] args) {{
        var res = {fn_name}({input_str});
        var expected = {output};
        if (res != expected) {{
            throw new RuntimeException("Test failed");
        }} else {{
            System.out.println("Test passed");
        }}
    }}
}}
'''
    elif language == "typescript":
        full_code = f'''
{function}
const res = {fn_name}({input_str});
const expected = {output};
if (res !== expected) {{
    throw new Error(`Test failed`);
}}
'''
    elif language == "csharp":
        full_code = f'''
using System;

public class Test{fn_name} {{
    {function}
    public static void Main(string[] args) {{
        var res = {fn_name}({input_str});
        var expected = {output};
        if (res != expected) {{
            throw new Exception("Test failed");
        }} else {{
            Console.WriteLine("Test passed");
        }}
    }}
}}
'''
    elif language == "rust":
        full_code = f'''
{function}
fn main() {{
    let res = {fn_name}({input_str});
    let expected = {output};
    if res != expected {{
        panic!("Test failed");
    }} else {{
        println!("Test passed");
    }}
}}
'''
    elif language == "php":
        full_code = f'''
<?php
{function}
$res = {fn_name}({input_str});
$expected = {output};
if ($res != $expected) {{
    throw new Exception("Test failed");
}}
?>
'''
    elif language == "bash":
        input_str = ' '.join([f'"{i}"' if type(i) is str else str(i) for i in input])
        full_code = f'''
{function}
res=$({fn_name} {input_str})
expected={output}
if [ "$res" != "$expected" ]; then
    echo "Test failed, res: $res, expected: $expected" 1>&2
    exit 1
fi
'''
    elif language == "ruby":
        full_code = f'''
{function}
res = {fn_name}({input_str})
expected = {output}
if res != expected
    raise "Test failed"
end
'''
    elif language == "perl":
        full_code = f'''
{function}
$res = {fn_name}({input_str});
$expected = {output};
if ($res != $expected) {{
    die "Test failed";
}}
'''
    elif language == "scala":
        full_code = f'''
object Test{fn_name} {{
    {function}
    def main(args: Array[String]): Unit = {{
        val res = {fn_name}({input_str})
        val expected = {output}
        if (res != expected) {{
            throw new Exception(s"Test failed: Expected $expected, got $res")
        }} else {{
            println("Test passed")
        }}
    }}
}}
'''
    elif language == "swift":
        # For Swift, we need to add parameter labels in function call
        # Convert "a, b" to "a: a, b: b" format
        swift_input_args_str = re.findall(f'func {fn_name}\((.*)\)', function)[0]
        swift_input_args = re.findall(r'(\w+):', swift_input_args_str)
        swift_input_params = []
        for arg, value in zip(swift_input_args, input):
            if isinstance(value, str):
                value = f'"{value}"'
            if len(input) > 1:
                swift_input_params.append(f'{arg}: {value}')
            else:
                swift_input_params.append(f'{value}')
        swift_input_str = ', '.join(swift_input_params)

        
        full_code = f'''
{function}

let res = {fn_name}({swift_input_str})
let expected = {output}
if res != expected {{
    fatalError("Test failed: Expected \\(expected), got \\(res)")
}} else {{
    print("Test passed")
}}
'''
    return full_code

async def check_function_call_test_cases_parallel(code: str,
                                                  cases: List[Dict[str, Any]],
                                                  config: TestConfig) -> List[EvalTestCase]:
    """
    Given the function call input and output, append `assert fn(input) == output` to the code and run it.
    cases = [
        {'type': 'function_call', 'fn_name': 'sort_twisted37', 'input': [[1, 2, 3, 4, 5, 6, 7, 8, 9]], 'output': [[1, 2, 7, 4, 5, 6, 3, 8, 9]]}, 
        {'type': 'function_call', 'fn_name': 'sort_twisted37', 'input': [[12, 13, 14]], 'output': [[12, 14, 13]]}, 
        {'type': 'function_call', 'fn_name': 'sort_twisted37', 'input': [[9, 2, 4, 7, 3]], 'output': [[2, 7, 4, 3, 9]]}
    ]
    """
    instance_id = uuid.uuid4().hex
    instance_logger = logger.bind(instance_id=instance_id)
    result = []
    tasks: List[asyncio.Task[EvalTestCase]] = []

    check_function_call_test_case_limited = check_function_call_test_case
    if sandbox_config.dataset.max_runner_concurrency > 0:
        check_function_call_test_case_limited = max_concurrency(
            sandbox_config.dataset.max_runner_concurrency)(check_function_call_test_case)
            
    language = config.language
    for case in cases:
        fn_name = case['fn_name']
        input = case['input']
        output = case['output']
        full_code = concat_function_assertion(code, fn_name, input, output, language)
        task = asyncio.create_task(check_function_call_test_case_limited(full_code, case, config))
        tasks.append(task)
    
    run_all_cases = config.extra.get("run_all_cases", False)
    total_timeout = config.extra.get("total_timeout", 300)
    deadline = asyncio.get_running_loop().time() + total_timeout

    for task_idx, task in enumerate(tasks):
        instance_logger.info(f"check_function_call_test_cases_parallel, total_timeout: {total_timeout}, task_idx: {task_idx}, deadline: {deadline}, current_time: {asyncio.get_running_loop().time()}")
        if asyncio.get_running_loop().time() > deadline:
            instance_logger.error("check_function_call_test_cases_parallel timeout", total_timeout=total_timeout, task_idx=task_idx)
            for remaining_task in tasks:
                if not remaining_task.done():
                    time_out_outcome = EvalTestCase(passed=False, exec_info=RunCodeResponse(status=RunStatus.SandboxError, message= "Total Timeout"), test_info=None)
                    result.append(time_out_outcome)
                    remaining_task.cancel()
            break
        try:
            outcome = await task
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Failed to check function call test case: {e}')
        result.append(outcome)

        if not run_all_cases and not outcome.passed:
            for remaining_task in tasks:
                if not remaining_task.done():
                    remaining_task.cancel()
            break

    return result

def parse_jest_cases(report_data: str) -> List[Dict[str, Any]]:
    if isinstance(report_data, str):
        report = json.loads(report_data)
    else:
        report = report_data

    test_cases = []

    for test_suite in report['testResults']:
        file_path = test_suite['testFilePath']

        for test_case in test_suite['testResults']:
            result = {
                'passed': test_case['status'] == 'passed',
                'full_name': test_case['fullName'],
                'file': file_path,
                'suite': ' > '.join(test_case['ancestorTitles']),
                'test': test_case['title'],
                'failure_messages': test_case['failureMessages']
            }
            test_cases.append(result)

    return test_cases

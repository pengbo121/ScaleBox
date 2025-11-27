import argparse
import json
import os
import sys
import time
import random
import threading
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import requests

# Optional imports with graceful fallbacks
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from tqdm import tqdm
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

INSTRUCTION = """
Here is an example of task description and a special judge program in Python.
<problem>
CERC 2024 in Wroclaw is approaching, and all hands are on deck to help with organization. Even Wroclaw’s dwarfs are helping! As a member of the organizing committee, each dwarf was given a plaque, so everyone can identify him. Dwarfs are very creative creatures, so each of them came up with an unique nickname that they want to be displayed on the plaque. A nickname is correct if its first four letters match with first four letters of the dwarf’s name (we treat lowercase and uppercase letters as different). For example, dwarf Mathew can have a plaque that reads Mathy, but he can not have a plaque that reads Matty or MATHY.  

Dwarf the Sloppy printed the plaques. Since he is, well, sloppy, he made no notes which plaque is whose. To make things worse, some of the plaques may contain errors. Help Sloppy to figure out which plaque belongs to whom: You will be given the list of all dwarfs’ names and the list of all nicknames on the plaques. Write a program which decides whether there exists an assignment of the plaques to the names such that the plaque contains a proper nickname of the assigned name. If such an assignment exists, your program should also print it.  

 Input  
The first line of the input contains a single integer $N$, the number of dwarfs. Each of the following $N$ lines contains a single dwarf’s name, which is a string of lowercase and uppercase English letters.  

Each of the next $N$ lines contains a single nickname written on a plaque, which is a string of lowercase and uppercase English letters.  


 Limits  
$1 \\leq N \\leq 100000$, each name and nickname contains at least 4 and at most 400000 letters, the sum of all lengths of the names and the sum of all lengths of nicknames does not exceed 400000, it is guaranteed that no name and no nickname appears twice.  


Output  
The first line of output should contain a single word – YES if the assignment described above is possible and NO if there is no such an assignment.  

If the answer is YES, then the following $N$ lines should contain the correct assignment: Each of the lines should contain the name of the dwarf and the nickname assigned to him, separated by a single space. If there are many possible assignments, print any of them.  


Examples  
standard input
4
Slopy
Mathy
Thinky
Cody
Thinky
Math
Slopppy
Codythesecond
standard output
YES
Cody Codythesecond
Mathy Math
Slopy Slopppy
Thinky Thinky
standard input
3
Writy
Buggy
Solvy
Bogg
Write
Solvy
standard output
NO
</problem>

<special_judge_program>
```python
import sys
from typing import List


def read_file(filepath: str) -> List[str]:
	with open(filepath, 'r') as f:
		return f.read().strip().split('\n')


def validate_solution(stdin_path: str, jury_stdout_path: str, participant_output_path: str) -> bool:
	# Read files via read_file as per the required interface
	ans_lines = read_file(jury_stdout_path)
	ouf_lines = read_file(participant_output_path)
	# Tokenize by whitespace across all lines
	ans_tokens = (" ".join(ans_lines)).split()
	ouf_tokens = (" ".join(ouf_lines)).split()

	if not ans_tokens:
		# Jury output must contain at least one token (YES/NO)
		return False
	if not ouf_tokens:
		# Participant must also produce at least one token
		return False

	j = ans_tokens[0]
	p = ouf_tokens[0]

	if j != p:
		# Expected YES/NO must match exactly
		return False
	if j == "NO":
		return True

	# j == "YES": need to validate the assignment using stdin
	stdin_lines = read_file(stdin_path)
	in_tokens = (" ".join(stdin_lines)).split()
	it = 0

	# Read n
	if it >= len(in_tokens):
		return False
	try:
		n = int(in_tokens[it])
	except ValueError:
		return False
	it += 1
	if n < 0:
		return False

	# Read names (n tokens)
	if it + n > len(in_tokens):
		return False
	names = set(in_tokens[it:it + n])
	if len(names) != n:
		# While input guarantees uniqueness, reject malformed input
		return False
	it += n

	# Read nicknames (n tokens)
	if it + n > len(in_tokens):
		return False
	nicknames = set(in_tokens[it:it + n])
	if len(nicknames) != n:
		return False
	it += n

	# Now parse participant mapping pairs from ouf_tokens after the first token
	pos = 1  # skip initial YES
	need = 2 * n
	if pos + need > len(ouf_tokens):
		# Not enough tokens to provide n pairs
		return False

	used_names = set()
	used_nicks = set()

	for _ in range(n):
		name = ouf_tokens[pos]
		nick = ouf_tokens[pos + 1]
		pos += 2

		if name not in names:
			return False
		if nick not in nicknames:
			return False
		# Validate first 4 letters match (strings in inputs are guaranteed >= 4)
		if len(name) < 4 or len(nick) < 4:
			return False
		if name[:4] != nick[:4]:
			return False

		used_names.add(name)
		used_nicks.add(nick)

	# Must cover all without duplicates
	if len(used_names) != n or len(used_nicks) != n:
		return False

	# Do NOT enforce EOF or absence of extra tokens to mirror the C++ judge semantics
	return True
# Default paths and execution style aligned with the example snippet
stdin_path = "stdin.txt"
jury_stdout_path = "stdout.txt"
participant_output_path = "answer.txt"

_ok = validate_solution(stdin_path, jury_stdout_path, participant_output_path)
if _ok:
	sys.exit(0)
else:
	sys.exit(1)
```
</special_judge_program>
-----

Given the stdin, stdout, answer and programming task, write the JUDGE python program to check if the answer is as valid as the stdout. You may want to leverage stdout to save computation when needed.
```python
import sys

def read_file(filepath):
    with open(filepath, 'r') as f:
        return f.read().strip().split('\n')


def validate_solution(stdin_path, stdout_path, answer_path):
    stdin_lines = read_file(stdin_path)
    stdout_lines = read_file(stdout_path)
    participant_output = read_file(answer_path)

    if participant_output == [''] and stdout_lines != ['']:
        return False
    # ...


stdin_path = "stdin.txt"
stdout_path = "stdout.txt"
answer_path = "answer.txt"

is_valid = validate_solution(stdin_path, stdout_path, answer_path)

if is_valid:
    sys.exit(0)
else:
    sys.exit(1)
```
"""

def parse_args():
    p = argparse.ArgumentParser(description="Generate special judge program (parallel single-item LLM calls).")
    p.add_argument("--sandbox_url", type=str)
    p.add_argument("--api_key", type=str)
    p.add_argument("--base_url", type=str, default="https://api.deepseek.com", help="Optional OpenAI-compatible base URL.")
    p.add_argument("--model", type=str, default="deepseek-chat")
    p.add_argument("--data_path", type=str, default="data/require_special_judge.parquet", help="HF dataset name or local JSON/JSONL file.")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--text_field", type=str, default="prompt", help="Field containing problem text.")
    p.add_argument("--id_field", type=str, default="problem_id", help="Optional id field to persist.")
    p.add_argument("--batch_size", type=int, default=8, help="Alias of --concurrency if --concurrency omitted.")
    p.add_argument("--concurrency", type=int, default=None, help="Parallel worker count.")
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_path", type=str, default="data/special_judge_deepseek-chat.jsonl", help="Output JSONL file.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--rate_limit_sleep", type=float, default=5.0)
    p.add_argument("--max_retries", type=int, default=6)
    p.add_argument("--max_rewrite", type=int, default=1)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--run_timeout", type=int, default=30)
    return p.parse_args()

def load_input_dataset(path: str, split: str, text_field: Optional[str], max_items: Optional[int]) -> (List[Dict[str, Any]], str):
    data: List[Dict[str, Any]] = []
    if os.path.isfile(path):
        if path.endswith(".jsonl"):
            data = load_dataset("json", data_files=path, split="train")
        elif path.endswith(".parquet"):
            data = load_dataset("parquet", data_files=path, split="train")
        else:
            raise ValueError("Unsupported file extension (use .json or .jsonl).")
        data = list(data)
    else:
        if load_dataset is None:
            raise RuntimeError("datasets not installed.")
        ds = load_dataset(path, split=split)
        data = list(ds)
    
    print(f"Loaded {len(data)} items from {path} split={split}", file=sys.stderr)

    if max_items is not None:
        data = data[:max_items]

    if not data:
        raise ValueError("No data loaded.")

    if text_field is None:
        candidates = ["problem", "prompt", "description", "statement", "question", "text", "content"]
        for c in candidates:
            if c in data[0]:
                text_field = c
                break
        if text_field is None:
            raise ValueError("Could not infer --text_field.")
    return data, text_field

def already_processed_ids(output_path: str, id_field: Optional[str]) -> set:
    done = set()
    if not os.path.isfile(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if id_field and id_field in obj:
                    done.add(obj[id_field])
                else:
                    done.add(obj.get("_global_index"))
            except:
                continue
    return done

def build_user_prompt(item: Dict[str, Any], text: str) -> str:
    snippet = text.strip()
    if len(snippet) > 12000:
        snippet = snippet[:12000] + "...[TRUNCATED]\n" + snippet[-1000:]
    return f"<problem_that_require_special_judging>\n{snippet}\n</problem_that_require_special_judging>\n\nNow, write the python code for judging the correctness of the answer output.\n"

def evaluate(completion, test_cases, special_judge_program, max_retries=3, retry_delay=2):
    # Remove ".*</think>" 
    # completion = re.sub(r'.*</think>', '', completion, flags=re.DOTALL).strip()
    language = 'python'

    payload = {
        "completion": completion,
        "config": {
            "language": language,
            "compile_timeout": args.run_timeout,
            "run_timeout": args.run_timeout,
            "provided_data": test_cases,
            "extra": {
                "run_all_cases": False, # True,
                "total_timeout": 30,
                "special_judge_program": special_judge_program,
                "special_judge_language": language,
                "force_special_judge": True
            }
        }
    }

    for retry_count in range(max_retries):
        try:
            response = requests.post(f'{args.sandbox_url}/common_evaluate', json=payload, timeout=60)
            result = response.json()
            if response.status_code == 200:
                result['language'] = language
                return result
            else:
                return {"error": f'API responded with code {response.status_code}: {response.text}', "language": language}
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries-1:
                print(f"Failed after {max_retries} retries: {str(e)}", file=sys.stderr)
            wait_time = retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
            print(f"Attempt {retry_count} failed: {str(e)}. Retrying in {wait_time}s...", file=sys.stderr)
            time.sleep(wait_time)
    return {"error": "Max retries exceeded", "language": language}


def write_code(item: Dict[str, Any],
                  text_field: str,
                  client_params: Dict[str, Any],
                  model: str,
                  max_retries: int,
                  rate_limit_sleep: float,
                  refine: bool = False,
                  previous_code: str = "",
                  dry_run: bool = False) -> Dict[str, Any]:
    text = item[text_field][0]["content"]

    # Instantiate client inside thread (safer for some SDKs)
    client = OpenAI(**client_params)

    delay = 1.0
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": build_user_prompt(item, text)},
    ],
    if refine:
        messages.append({"role": "assistant", "content": previous_code})
        messages.append({"role": "user", "content": "The above code is incorrect. Please refine it based on the problem description."})

    for attempt in range(max_retries):
        print(f"Item {item['_global_index']} attempt {attempt+1}/{max_retries}", file=sys.stderr)
        try:
            # Streaming response
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": build_user_prompt(item, text)},
                ],
                temperature=0.7,
                max_completion_tokens=16384,
                stream=True,
            )
            parts = []
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    reasoning_content = getattr(delta, "reasoning_content", None)
                    print(content or reasoning_content, end="")
                    if content:
                        parts.append(content)
                except Exception:
                    continue
            raw = "".join(parts)
            print(f"Item {item['_global_index']} LLM infer:\n{raw}")
            return parse_code(raw)
        except RateLimitError:
            print(f"Item {item['_global_index']} rate limited, retrying...", file=sys.stderr)
            sleep_for = rate_limit_sleep * (2 ** attempt) + random.random()
            time.sleep(sleep_for)
        except APITimeoutError:
            print(f"Item {item['_global_index']} API timeout, retrying...", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
        except APIError as e:
            print(f"Item {item['_global_index']} API error: {e}, retrying...", file=sys.stderr)
            sc = getattr(e, "status_code", 500)
            if sc >= 500:
                time.sleep(delay)
                delay *= 2
            else:
                # Non-retryable
                break
        except Exception as e:
            print(f"Item {item['_global_index']} unexpected error: {e}, retrying...", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    # Fallback result
    return {
        "needs_special_judge": False,
        "categories": [],
        "reason": "failed to generate",
        "confidence": 0.0
    }

def parse_code(raw: str) -> str:
    try:
        raw = re.sub(r".*</think>", "", raw, flags=re.DOTALL).strip()
        match = re.search(r"```(?:python)?\n(.*?)```", raw, flags=re.DOTALL)
        if match:
            code = match.group(1).strip()
            print(code)
    except:
        code = ""
    return code

def main():
    random.seed(args.seed)

    client_params = {}
    client_params["api_key"] = args.api_key
    if args.base_url:
        client_params["base_url"] = args.base_url
    client = OpenAI(**client_params)


    data, text_field = load_input_dataset(args.data_path, args.split, args.text_field, args.max_items)

    for i, item in enumerate(data):
        item["_global_index"] = i

    if args.resume:
        done_ids = already_processed_ids(args.output_path, args.id_field)
        print(f"Resuming, found {len(done_ids)} already processed items.", file=sys.stderr)
    else:
        done_ids = set()

    to_generate: List[Dict[str, Any]] = []
    for item in data:
        key = item.get(args.id_field) if args.id_field else item["_global_index"]
        if key not in done_ids:
            to_generate.append(item)
    #         print(item['prompt'])
    #         print(item['in_source_id'])
    #         print([item['attempt_code']])
    # exit(0)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    out_f = open(args.output_path, "a", encoding="utf-8")
    write_lock = threading.Lock()

    total = len(data)
    print(f"Total: {total}. Pending: {len(to_generate)}. Concurrency: {args.concurrency or args.batch_size}", file=sys.stderr)

    client_params = {}
    client_params["api_key"] = args.api_key
    if args.base_url:
        client_params["base_url"] = args.base_url
    client = OpenAI(**client_params)

    concurrency = args.concurrency or args.batch_size
    pbar = tqdm(total=len(to_generate), desc="Generating")

    def task(item: Dict[str, Any]):
        out_obj = dict(item)
        empty_result = {
            "special_judge_program": "",
            "special_judge_evaluation": {},
            "accepted": False
        }
        if "gold_standard_solution" not in item:
            out_obj.update(empty_result)
        else:
            completion = item["gold_standard_solution"]
            test_cases = eval(item.get("verification_info", ""))
            # test_cases = test_cases.get("test_cases", [])

            for _ in range(args.max_rewrite):
                special_judge_program = write_code(item,
                                    text_field=text_field,
                                    client_params=client_params,
                                    model=args.model,
                                    max_retries=args.max_retries,
                                    rate_limit_sleep=args.rate_limit_sleep,
                                    dry_run=args.dry_run)
                if special_judge_program.strip() == "":
                    print(f"Item {item['_global_index']} got empty special judge program, retrying...", file=sys.stderr)
                    continue
                
                def eval_spj(completion, test_cases, special_judge_program):
                                    
                    eval_result = evaluate(completion, test_cases, special_judge_program, max_retries=3, retry_delay=2)
                    print("gold standard eval result:", eval_result)
                    accepted = eval_result.get("accepted", False)
                    
                    empty_completion = "```python\ndef empty_judge():\n    pass\n```"
                    eval_empty = evaluate(empty_completion, test_cases, special_judge_program, max_retries=3, retry_delay=2)
                    empty_accepted = eval_empty.get("accepted", True)
                    print("empty completion eval result:", eval_empty)

                    return accepted, empty_accepted, eval_result
                
                accepted, empty_accepted, eval_result = eval_spj(completion, test_cases, special_judge_program)
                if accepted and (not empty_accepted):
                    break
                # else:
                #     special_judge_program = write_code(item,
                #                         text_field=text_field,
                #                         client_params=client_params,
                #                         model=args.model,
                #                         max_retries=args.max_retries,
                #                         rate_limit_sleep=args.rate_limit_sleep,
                #                         refine=True,
                #                         previous_code=special_judge_program,
                #                         dry_run=args.dry_run)
                #     accepted, empty_accepted, eval_result = eval_spj(completion, test_cases, special_judge_program)
                #     if accepted and (not empty_accepted):
                #         break

                print(f"Item {item['_global_index']} evaluation not accepted, retrying special judge generation...", file=sys.stderr)

            # Merge & write
            out_obj.update({
                "special_judge_program": special_judge_program,
                "special_judge_evaluation": eval_result,
                "accepted": (accepted) and (not empty_accepted)
            })
            print(f"Item {item['_global_index']} special judge evaluation: {eval_result}", file=sys.stderr)
        if out_obj["accepted"]:
            with write_lock:
                out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                out_f.flush()
                pbar.update(1)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(task, item) for item in to_generate]
        for _ in as_completed(futures):
            pass  # progress handled in task

    pbar.close()
    out_f.close()
    print("Done.", file=sys.stderr)

if __name__ == "__main__":
    args = parse_args()
    main()

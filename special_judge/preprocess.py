import json
import os
from datasets import load_dataset


ds = load_dataset("PrimeIntellect/verifiable-coding-problems", split="train", trust_remote_code=True)
print(ds[0].keys())

import ast

def change_test_list_format(old_tests_list):
    """
    Convert 
    [{'type': 'function_call', 'fn_name': 'sort_twisted37', 'input': [[1, 2, 3, 4, 5, 6, 7, 8, 9]], 'output': [[1, 2, 7, 4, 5, 6, 3, 8, 9]]}, {'type': 'function_call', 'fn_name': 'sort_twisted37', 'input': [[12, 13, 14]], 'output': [[12, 14, 13]]}, {'type': 'function_call', 'fn_name': 'sort_twisted37', 'input': [[9, 2, 4, 7, 3]], 'output': [[2, 7, 4, 3, 9]]}]
    into
    {'fn_name': 'findNumber', 'input': [[1], [5], [10], [9], [7], [1], [2], [3], [4], [11], [1000000000000], [999999999999]], 'output': [['1'], ['9'], ['19'], ['17'], ['13'], ['1'], ['3'], ['5'], ['7'], ['31'], ['113559777777777779'], ['113559777777777777']]}
    """
    new_tests = {
        "input": [],
        "output": []
    }
    if "fn_name" in old_tests_list[0]:
        new_tests["fn_name"] = old_tests_list[0]["fn_name"]
        new_tests["type"] = "function_call"
        if type(old_tests_list[0]["input"]) == str and type(old_tests_list[0]["input"][0]) == str:
            print()
            for test in old_tests_list:
                new_tests["input"].append([json.loads(the_input) for the_input in test["input"].split("\n")])
                new_tests["output"].append(json.loads(test["output"]))
    else:
        new_tests["fn_name"] = None
        new_tests["type"] = "stdin_stdout"
        
    for test in old_tests_list:
        new_tests["input"].append(test["input"])
        new_tests["output"].append(test["output"])
    return new_tests

def process_entry(entry):
    conversation = [
        {"role": "user", "content": entry["prompt"]},
    ]
    empty_data = {
        "prompt": conversation,
        "solutions": [""],
        "reward_model": {"ground_truth": "", "style": "rule"},
        "data_source": entry["source"]
    }
    gold_standard_solution = entry["gold_standard_solution"]
    if gold_standard_solution is None:
        return empty_data
    if gold_standard_solution.startswith("```python") and gold_standard_solution.endswith("```"):
        tests = entry["verification_info"]
        if isinstance(tests, str):
            try:
                tests = ast.literal_eval(tests)
            except (ValueError, SyntaxError):
                try:
                    tests = json.loads(entry["verification_info"])
                except (json.JSONDecodeError, SyntaxError, ValueError) as e:
                    print(repr(entry["verification_info"]))
                    print(f"Error in json.loads: {e}")
                    return empty_data
        if not isinstance(tests, dict):
            return empty_data
        language = tests.get("language", "")
        if language != "python":
            return empty_data
        tests_list = tests.get("test_cases", [])
        if len(tests_list) <= 4:
            return empty_data
        if not (isinstance(tests_list, list) and "input" in tests_list[0] and "output" in tests_list[0]):
            return empty_data
        tests_list = change_test_list_format(tests_list)
        return {
            "prompt": conversation,
            "solutions": [gold_standard_solution],
            "reward_model": {"ground_truth": json.dumps(tests_list), "style": "rule"},
            "data_source": entry["source"]
        }
    return empty_data

processed_ds = ds.map(process_entry, remove_columns=ds.column_names)
print("Processed dataset size:", len(processed_ds))
processed_ds = processed_ds.filter(lambda x: x["solutions"] != [""])
print("Filtered dataset size:", len(processed_ds))
print(processed_ds[0])

processed_ds.to_parquet("data/PrimeIntellect-verifiable-coding-problems-python.parquet")

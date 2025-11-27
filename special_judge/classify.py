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

# Optional imports with graceful fallbacks
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from tqdm import tqdm
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

INSTRUCTION = """You are an assistant that classifies programming / algorithm / data processing tasks regarding SPECIAL JUDGE need.

Decide for the given task text:
1. multiple_solutions? For example:
 - If there are multiple answers, print any of them
 - If there are multiple solutions, you are allowed to print any of them.
 - If there are multiple possible solutions, print any of them.
 - If there are different possible orders with a correct answer, print any of them.
 - If there are multiple solutions, satisfying the problem condition(s), you can print any "one" solution.

2. float_comparison? (floating point answers, precision, tolerance, absolute/relative error, decimals)

Return JSON object:
{
 "reason": "<short justification <less than 160 words>",
 "needs_special_judge": <true|false>,
 "categories": [ zero or more of "multiple_solutions","float_comparison" ],
 "confidence": <float 0..1>
}

Rules:
 - needs_special_judge is true iff categories non-empty.
 - confidence: 0.9 clear indicators, 0.6 somewhat, 0.3 guess.
 - Keep output STRICT JSON. No extra keys.
"""

def parse_args():
    p = argparse.ArgumentParser(description="Classify dataset items for special judge requirements (parallel single-item LLM calls).")
    p.add_argument("--api_key", type=str)
    p.add_argument("--base_url", type=str, default="https://api.deepseek.com", help="Optional OpenAI-compatible base URL.")
    p.add_argument("--model", type=str, default="deepseek-chat")
    p.add_argument("--data_path", type=str, default="PrimeIntellect/verifiable-coding-problems", help="HF dataset name or local PARQUET/JSONL file.")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--text_field", type=str, default="prompt", help="Field containing problem text.")
    p.add_argument("--id_field", type=str, default="problem_id", help="Optional id field to persist.")
    p.add_argument("--batch_size", type=int, default=16, help="Alias of --concurrency if --concurrency omitted.")
    p.add_argument("--concurrency", type=int, default=None, help="Parallel worker count.")
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_path", type=str, default="data/classified_deepseek-chat.jsonl", help="Output JSONL file.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--rate_limit_sleep", type=float, default=5.0)
    p.add_argument("--max_retries", type=int, default=6)
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
    if len(snippet) > 8000:
        snippet = snippet[:8000] + "...[TRUNCATED]\n" + snippet[-1000:]
    return f"<problem_to_be_classified>\n{snippet}\n</problem_to_be_classified>"

def classify_item(item: Dict[str, Any],
                  text_field: str,
                  client_params: Dict[str, Any],
                  model: str,
                  max_retries: int,
                  rate_limit_sleep: float) -> Dict[str, Any]:
    text = item[text_field][0]["content"]

    # Instantiate client inside thread (safer for some SDKs)
    client = OpenAI(**client_params)

    delay = 1.0
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
                max_completion_tokens=8192,
                stream=True,
            )
            parts = []
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    print(content, end="")
                    if content:
                        parts.append(content)
                except Exception:
                    continue
            raw = "".join(parts)
            print(f"Item {item['_global_index']} LLM infer:\n{raw}")
            return parse_single_json(raw)
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
        "reason": "failed to classify",
        "confidence": 0.0
    }

def parse_single_json(raw: str) -> Dict[str, Any]:
    try:
        raw = re.sub(r".*</think>", "", raw, flags=re.DOTALL).strip()
        if raw.startswith("```json"):
            raw = raw[7:-3].strip()
        elif raw.startswith("```"):
            raw = raw[3:-3].strip()
        obj = json.loads(raw)
        # Basic normalization
        obj["needs_special_judge"] = bool(obj.get("needs_special_judge") or obj.get("categories"))
        obj["categories"] = obj.get("categories", [])
        obj["reason"] = obj.get("reason", "")
        obj["confidence"] = obj.get("confidence", None)
        return obj
    except json.JSONDecodeError as e:
        raise ValueError(f"Bad JSON: {e} - raw: {raw[:200]}")

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
    else:
        done_ids = set()

    to_classify: List[Dict[str, Any]] = []
    for item in data:
        key = item.get(args.id_field) if args.id_field else item["_global_index"]
        if key not in done_ids:
            to_classify.append(item)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    out_f = open(args.output_path, "a", encoding="utf-8")
    write_lock = threading.Lock()

    total = len(data)
    print(f"Total: {total}. Pending: {len(to_classify)}. Concurrency: {args.concurrency or args.batch_size}", file=sys.stderr)

    client_params = {}
    client_params["api_key"] = args.api_key
    if args.base_url:
        client_params["base_url"] = args.base_url
    client = OpenAI(**client_params)

    concurrency = args.concurrency or args.batch_size
    pbar = tqdm(total=len(to_classify), desc="Classifying")

    def task(item: Dict[str, Any]):
        result = classify_item(item,
                               text_field=text_field,
                               client_params=client_params,
                               model=args.model,
                               max_retries=args.max_retries,
                               rate_limit_sleep=args.rate_limit_sleep)
        # Merge & write
        out_obj = dict(item)
        out_obj.update({
            "needs_special_judge": result.get("needs_special_judge", False),
            "categories": result.get("categories", []),
            "reason": result.get("reason", ""),
            "confidence": result.get("confidence", None),
        })
        print(f"Item {item['_global_index']} classified as: {out_obj['categories']} (confidence {out_obj['confidence']})", file=sys.stderr)
        with write_lock:
            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_f.flush()
            pbar.update(1)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(task, item) for item in to_classify]
        for _ in as_completed(futures):
            pass  # progress handled in task

    pbar.close()
    out_f.close()
    print("Done.", file=sys.stderr)

if __name__ == "__main__":
    args = parse_args()
    main()

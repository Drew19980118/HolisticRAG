import json
import re
import argparse
from typing import List, Union

def normalize_text(s: str) -> str:
    """Lowercase, remove punctuation and extra spaces, for F1 tokenization (EM can also compare normalized strings)"""
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    return s

def tokenize(s: str) -> List[str]:
    """Split string into tokens by whitespace"""
    return s.split()

def em_score(pred: str, true: str) -> int:
    """Exact Match: 1 if normalized strings are identical, else 0"""
    pred_norm = normalize_text(pred)
    true_norm = normalize_text(true)
    return 1 if pred_norm == true_norm else 0

def f1_score(pred: str, true: str) -> float:
    """Token-level F1 score"""
    pred_tokens = tokenize(normalize_text(pred))
    true_tokens = tokenize(normalize_text(true))
    if not pred_tokens and not true_tokens:
        return 1.0
    if not pred_tokens or not true_tokens:
        return 0.0
    common = set(pred_tokens) & set(true_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(true_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def best_metrics(pred: str, gold_answers: Union[str, List[str]]) -> tuple:
    """
    For a single prediction and a set of gold answers (string or list of strings),
    return the best EM and F1 across all gold answers.
    """
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]
    best_em = 0
    best_f1 = 0.0
    for ans in gold_answers:
        em = em_score(pred, ans)
        f1 = f1_score(pred, ans)
        if em > best_em:
            best_em = em
        if f1 > best_f1:
            best_f1 = f1
        if best_em == 1 and best_f1 == 1.0:
            break
    return best_em, best_f1

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions against gold answers for a given benchmark."
    )
    parser.add_argument("--benchmark", "-b", type=str, required=True,
                        help="Benchmark name, e.g., 'musique', 'hotpotqa'. "
                             "Will read predictions from data/{benchmark}/{benchmark}_results.jsonl "
                             "and gold answers from data/{benchmark}/{benchmark}_gold.json")
    parser.add_argument("--pred-file", type=str, default=None,
                        help="Override path to predictions JSONL file (default: built from benchmark)")
    parser.add_argument("--gold-file", type=str, default=None,
                        help="Override path to gold answers JSON file (default: built from benchmark)")
    args = parser.parse_args()

    # Build default paths
    if args.pred_file is None:
        pred_file = f"data/{args.benchmark}/{args.benchmark}_results.jsonl"
    else:
        pred_file = args.pred_file

    if args.gold_file is None:
        gold_file = f"data/{args.benchmark}/{args.benchmark}.json"
    else:
        gold_file = args.gold_file

    print(f"Loading predictions from: {pred_file}")
    print(f"Loading gold answers from: {gold_file}")

    # 1. Load gold answers (JSON array)
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    gold_map = {}
    for item in gold_data:
        q = item.get("question")
        ans = item.get("answer")
        if q is None or ans is None:
            continue
        gold_map[q] = ans

    # 2. Load predictions (JSONL)
    predictions = []
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pred = json.loads(line)
            predictions.append(pred)

    total_em = 0
    total_f1 = 0.0
    count = 0
    for pred in predictions:
        q = pred.get("original_query")
        pred_ans = pred.get("final_answer")
        if q is None or pred_ans is None:
            print(f"Warning: prediction missing 'original_query' or 'final_answer', skipping: {pred}")
            continue
        gold_ans = gold_map.get(q)
        if gold_ans is None:
            print(f"Warning: gold answer not found for query: {q}")
            continue

        em, f1 = best_metrics(pred_ans, gold_ans)
        total_em += em
        total_f1 += f1
        count += 1
        print(f"Query: {q[:80]}...")
        print(f"  Prediction: {pred_ans}")
        print(f"  Gold: {gold_ans}")
        print(f"  EM: {em}, F1: {f1:.4f}\n")

    if count == 0:
        print("No queries evaluated.")
        return

    avg_em = total_em / count
    avg_f1 = total_f1 / count
    print(f"Evaluated {count} queries")
    print(f"Average EM: {avg_em:.4f}")
    print(f"Average F1: {avg_f1:.4f}")

if __name__ == "__main__":
    main()
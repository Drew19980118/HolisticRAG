import requests
import time
import json
import re
import argparse
from typing import List, Dict, Any
from collections import defaultdict
from prompts.NER import NER_SYSTEM, NER_ONE_SHOT_PARAGRAPH, NER_ONE_SHOT_OUTPUT
from prompts.relation_extraction import (
    RELATION_SYSTEM,
    RELATION_EXAMPLE_PASSAGE,
    RELATION_EXAMPLE_OUTPUT,
    RELATION_USER_TEMPLATE
)
from prompts.sub_rel_matching import SUBJECT_RELATION_SYSTEM, SUBJECT_RELATION_USER_TEMPLATE
from prompts.triple_completion import OBJECT_COMPLETION_SYSTEM, OBJECT_COMPLETION_USER_TEMPLATE

# ==================== API call function ====================
def query_vllm_messages(messages: List[Dict[str, str]], model_name: str = "Qwen/Qwen2.5-32B-Instruct",
                        max_retries: int = 3) -> str:
    """
    Query the vLLM API with a list of messages. Supports retries.
    """
    url = "http://localhost:8000/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "seed": None,
        "temperature": 0
    }
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"  Attempt {attempt + 1} failed: HTTP {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        except requests.exceptions.Timeout:
            print(f"  Attempt {attempt + 1}: Request timeout")
            if attempt < max_retries - 1:
                time.sleep(5)
        except Exception as e:
            print(f"  Attempt {attempt + 1}: Request exception - {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    print("  All retries failed")
    return None


# ==================== JSON extraction function ====================
def extract_json(text: str) -> str:
    """Extract a JSON string from the model output."""
    json_block_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_block_match:
        return json_block_match.group(1).strip()
    matches = re.findall(r'(\{.*?\}|\[.*?\])', text, re.DOTALL)
    for match in reversed(matches):
        try:
            json.loads(match)
            return match.strip()
        except:
            continue
    return re.sub(r'```[a-z]*\s*|```', '', text).strip()


# ==================== Single paragraph processing ====================
def process_single_paragraph(text: str, idx: int, title: str, model_name: str) -> List[Dict[str, Any]]:
    """
    Process a single paragraph and return a list of triples (subject, relation, object).
    Returns an empty list if any step fails.
    """
    print(f"\n--- Processing paragraph {idx}: {title} ---")

    # 1. NER
    ner_messages = [
        {"role": "system", "content": NER_SYSTEM},
        {"role": "user", "content": NER_ONE_SHOT_PARAGRAPH},
        {"role": "assistant", "content": NER_ONE_SHOT_OUTPUT},
        {"role": "user", "content": text}
    ]
    ner_result = query_vllm_messages(ner_messages, model_name=model_name)
    if not ner_result:
        print(f"Paragraph {idx} NER failed, skipping")
        return []
    try:
        cleaned_ner = extract_json(ner_result)
        entities_dict = json.loads(cleaned_ner)
        entities = entities_dict.get("named_entities", [])
        entities = list(dict.fromkeys(entities))
    except Exception as e:
        print(f"Paragraph {idx} failed to parse NER result: {e}")
        return []

    if not entities:
        print(f"Paragraph {idx} no entities extracted, skipping")
        return []

    entities_str = json.dumps(entities, ensure_ascii=False)

    # 2. Relation extraction
    relation_messages = [
        {"role": "system", "content": RELATION_SYSTEM},
        {"role": "user", "content": RELATION_EXAMPLE_PASSAGE},
        {"role": "assistant", "content": RELATION_EXAMPLE_OUTPUT},
        {"role": "user", "content": RELATION_USER_TEMPLATE.format(passage=text, entities=entities_str)}
    ]
    relation_result = query_vllm_messages(relation_messages, model_name=model_name)
    if not relation_result:
        print(f"Paragraph {idx} relation extraction failed, skipping")
        return []
    try:
        cleaned_relation = extract_json(relation_result)
        relations_dict = json.loads(cleaned_relation)
        relations = relations_dict.get("relations", [])
        relations = list(dict.fromkeys(relations))
    except Exception as e:
        print(f"Paragraph {idx} failed to parse relation result: {e}")
        return []

    if not relations:
        print(f"Paragraph {idx} no relations extracted, skipping")
        return []

    # 3. Subject-Relation mapping
    relations_str = json.dumps(relations, ensure_ascii=False)
    subject_relation_messages = [
        {"role": "system", "content": SUBJECT_RELATION_SYSTEM},
        {"role": "user", "content": SUBJECT_RELATION_USER_TEMPLATE.format(
            passage=text, entities=entities_str, relations=relations_str
        )}
    ]
    subject_result = query_vllm_messages(subject_relation_messages, model_name=model_name)
    if not subject_result:
        print(f"Paragraph {idx} Subject-Relation mapping failed, skipping")
        return []
    try:
        cleaned_subject = extract_json(subject_result)
        subject_map = json.loads(cleaned_subject)
        for rel in subject_map:
            if isinstance(subject_map[rel], list):
                subject_map[rel] = list(dict.fromkeys(subject_map[rel]))
    except Exception as e:
        print(f"Paragraph {idx} failed to parse Subject-Relation result: {e}")
        return []

    # 4. Object completion (generate triples)
    mapping_str = json.dumps(subject_map, ensure_ascii=False, indent=2)
    object_messages = [
        {"role": "system", "content": OBJECT_COMPLETION_SYSTEM},
        {"role": "user", "content": OBJECT_COMPLETION_USER_TEMPLATE.format(
            passage=text,
            entities=json.dumps(entities, ensure_ascii=False),
            mapping=mapping_str
        )}
    ]
    triples_result = query_vllm_messages(object_messages, model_name=model_name)
    if not triples_result:
        print(f"Paragraph {idx} object completion failed, skipping")
        return []

    try:
        cleaned_triples = extract_json(triples_result)
        triples = json.loads(cleaned_triples)
        # Deduplicate based on subject, relation, object
        if isinstance(triples, list):
            seen = set()
            unique_triples = []
            for triple in triples:
                if isinstance(triple, dict) and all(k in triple for k in ('subject', 'relation', 'object')):
                    key = (triple['subject'], triple['relation'], triple['object'])
                    if key not in seen:
                        seen.add(key)
                        unique_triples.append(triple)
            triples = unique_triples
        else:
            triples = []
    except Exception as e:
        print(f"Paragraph {idx} failed to parse triples result: {e}")
        return []

    print(f"Paragraph {idx} completed, generated {len(triples)} triples")
    return triples


# ==================== Main: batch processing ====================
def main():
    parser = argparse.ArgumentParser(
        description="Extract knowledge triples (subject, relation, object) from a corpus using vLLM."
    )
    parser.add_argument("--benchmark", "-b", type=str, required=True,
                        help="Benchmark name, e.g., 'hotpotqa'. The script will read "
                             "data/{benchmark}/{benchmark}_corpus.json and write "
                             "data/{benchmark}/{benchmark}_corpus_triples.json")
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen2.5-32B-Instruct",
                        help="Model name to use for vLLM (default: Qwen/Qwen2.5-32B-Instruct)")
    args = parser.parse_args()

    # Build input and output paths from the benchmark name
    input_json_path = f"data/{args.benchmark}/{args.benchmark}_corpus.json"
    output_json_path = f"data/{args.benchmark}/{args.benchmark}_corpus_triples.json"

    print(f"Benchmark: {args.benchmark}")
    print(f"Model: {args.model}")
    print(f"Input file: {input_json_path}")
    print(f"Output file: {output_json_path}")

    # Read input JSON
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Input file format error: expected JSON array")
        return

    # Clear output file before starting (create empty file)
    with open(output_json_path, 'w', encoding='utf-8') as out_f:
        pass  # just create/truncate

    # Process each paragraph
    for idx, item in enumerate(data):
        if 'text' not in item:
            print(f"Warning: item {idx} missing 'text' field, skipping")
            continue

        text = item['text']
        title = item.get('title', f"Paragraph_{idx}")

        # Process paragraph with the specified model
        triples = process_single_paragraph(text, idx, title, model_name=args.model)

        # Build output object
        output_item = {
            "idx": idx,
            "title": title,
            "text": text,
            "triples": triples
        }

        # Append as a JSON line (JSONL format)
        with open(output_json_path, 'a', encoding='utf-8') as out_f:
            out_f.write(json.dumps(output_item, ensure_ascii=False) + '\n')

        print(f"Appended paragraph {idx} results to {output_json_path}")
        time.sleep(1)  # optional delay to avoid overwhelming the API

    print(f"\nAll processing completed! Results appended to {output_json_path}")

if __name__ == "__main__":
    main()
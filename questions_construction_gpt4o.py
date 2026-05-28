import json
import time
import argparse
import hashlib
import pandas as pd
from typing import List, Dict, Any
from prompts.question_construction import QUESTION_SYSTEM, QUESTION_USER_TEMPLATE
from embedding_model import get_embedding_model

# ==================== Helper functions ====================
def compute_mdhash_id(text: str, prefix: str = "question-") -> str:
    """Compute MD5 hash of the text and return with prefix."""
    md5_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    return f"{prefix}{md5_hash}"


def format_triple_from_dict(triple_dict: Dict[str, Any]) -> str:
    """Format a triple dict into a string (subject, relation, object)."""
    subj = triple_dict.get("subject", "")
    rel = triple_dict.get("relation", "")
    obj = triple_dict.get("object", "")
    return f"({subj}, {rel}, {obj})"

# ---------- Configuration (Please fill in your Azure OpenAI credentials) ----------
API_KEY = '[Your API Key]'
ENDPOINT = "[Your API Endpoint]"
headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

def query_gpt4o(messages: List[Dict[str, str]], model_name: str, max_retries: int = 3) -> Optional[str]:
    """
    Call the Azure OpenAI API with a list of messages (similar to OpenAI chat format).
    Supports retries. Returns the assistant's reply content or None on failure.
    """
    payload = {
        "messages": messages,
        "temperature": 1,
        "top_p": 0.95,
        "max_tokens": 400
    }
    # If the endpoint does not already include the deployment name, you may need to add it.
    # Usually the endpoint URL includes the deployment name, so we ignore model_name for now.
    # Alternatively, you can insert model_name into the payload as "model" if required.
    for attempt in range(max_retries):
        try:
            response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=120)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            else:
                print(f"   Attempt {attempt + 1} failed: HTTP {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        except requests.exceptions.Timeout:
            print(f"   Attempt {attempt + 1}: request timeout")
            if attempt < max_retries - 1:
                time.sleep(5)
        except Exception as e:
            print(f"   Attempt {attempt + 1}: request exception - {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    print("   All retries exhausted")
    return None


def generate_questions_for_triple(passage: str, triple: List[str], model_name: str) -> Dict[str, str]:
    """
    Generate two questions for a single triple: one with subject as answer, one with object.
    Returns dict: {"question_subject": "...", "question_object": "..."}
    """
    subj, pred, obj = triple[0], triple[1], triple[2]

    user_content = QUESTION_USER_TEMPLATE.format(passage=passage, subj=subj, pred=pred, obj=obj)

    messages = [
        {"role": "system", "content": QUESTION_SYSTEM},
        {"role": "user", "content": user_content}
    ]

    response = query_gpt4o(messages, model_name=model_name)
    if response is None:
        return {"question_subject": None, "question_object": None}

    # Parse response to extract Q1 and Q2
    q1, q2 = None, None
    lines = response.splitlines()
    for line in lines:
        if line.startswith("Q1:"):
            q1 = line[3:].strip()
        elif line.startswith("Q2:"):
            q2 = line[3:].strip()
    if not q1 or not q2:
        non_empty = [l.strip() for l in lines if l.strip() and not l.startswith(("Q1:", "Q2:"))]
        if len(non_empty) >= 2:
            q1, q2 = non_empty[0], non_empty[1]
        else:
            print(f"  Warning: Could not parse generated questions. Raw response: {response}")
            return {"question_subject": None, "question_object": None}

    return {"question_subject": q1, "question_object": q2}


def parse_json_objects(file_path: str) -> List[Dict]:
    """
    Parse a file containing multiple JSON objects (may be comma-separated, without outer array).
    Returns a list of dictionaries.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    objects = []
    decoder = json.JSONDecoder()
    idx = 0
    content_len = len(content)

    while idx < content_len:
        # Skip whitespace
        while idx < content_len and content[idx].isspace():
            idx += 1
        if idx >= content_len:
            break
        try:
            obj, end = decoder.raw_decode(content, idx)
            objects.append(obj)
            idx = end
            # Skip possible comma and whitespace
            while idx < content_len and (content[idx].isspace() or content[idx] == ','):
                idx += 1
        except json.JSONDecodeError as e:
            print(f"JSON parsing error at position {idx}: {e}")
            print("File may be incomplete, stopping parsing")
            break
    return objects


def generate_qa_pairs(input_path: str, output_path: str, model_name: str):
    """
    Read the triple corpus (JSON objects) and generate question pairs using vLLM.
    Output is JSONL: each line has "text", "triple", "questions".
    """
    docs = parse_json_objects(input_path)
    if not docs:
        raise ValueError("Failed to parse any document objects from input file")

    total_triples = 0
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for doc_idx, doc in enumerate(docs):
            passage = doc.get("text", doc.get("passage", ""))
            triples = doc.get("triples", [])
            if not triples:
                continue

            for triple_dict in triples:
                subj = triple_dict.get("subject", "")
                rel = triple_dict.get("relation", "")
                obj = triple_dict.get("object", "")

                print(f"Processing document {doc_idx + 1}/{len(docs)}, triple ({subj}, {rel}, {obj})")

                questions_full = generate_questions_for_triple(passage, [subj, rel, obj], model_name)
                q_sub = questions_full.get("question_subject")
                q_obj = questions_full.get("question_object")

                record = {
                    "text": passage,
                    "triple": {
                        "subject": subj,
                        "relation": rel,
                        "object": obj,
                    },
                    "questions": {
                        "question_subject": q_sub,
                        "question_object": q_obj
                    }
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                out_f.flush()
                total_triples += 1
                time.sleep(0.5)  # avoid overwhelming API

            print(f"Document {doc_idx + 1} completed, wrote {len(triples)} triples.")

    print(f"question generation finished. Total triples processed: {total_triples}. Output saved to {output_path}")


def compute_embeddings(input_jsonl: str, output_parquet: str, embedding_model_name: str, device: str):
    """
    Read the JSONL file containing question pairs, compute embeddings for each question,
    and save as Parquet with columns: content, hash_id, triple, passage_text, embedding.
    """
    # Initialize embedding model using factory
    print("Initializing embedding model...")
    embed_model = get_embedding_model(embedding_model_name, device=device)

    records = []      # each will have content, hash_id, triple, passage_text
    all_texts = []    # list of question texts for batch encoding

    print(f"Reading {input_jsonl}...")
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: line {line_num} JSON decode error, skipped - {e}")
                continue

            triple_dict = data.get("triple")
            questions_dict = data.get("questions")
            if not triple_dict or not questions_dict:
                print(f"Warning: line {line_num} missing 'triple' or 'questions', skipped")
                continue

            passage_text = data.get("text", "")
            triple_str = format_triple_from_dict(triple_dict)

            # Process both questions if not empty
            for q_key in ["question_subject", "question_object"]:
                question_text = questions_dict.get(q_key)
                if not question_text or not question_text.strip():
                    continue

                hash_id = compute_mdhash_id(question_text, prefix="question-")
                records.append({
                    "content": question_text,
                    "hash_id": hash_id,
                    "triple": triple_str,
                    "passage_text": passage_text
                })
                all_texts.append(question_text)

    print(f"Collected {len(records)} questions for embedding")

    if not records:
        print("No valid questions found, skipping embedding.")
        return

    # Compute embeddings in batches
    print("Computing embeddings...")
    embeddings = embed_model.batch_encode(all_texts)   # shape (n, dim)

    # Build DataFrame and save
    df = pd.DataFrame(records)
    df["embedding"] = list(embeddings)

    print(f"Saving to {output_parquet}...")
    df.to_parquet(output_parquet, index=False)
    print("Embedding computation completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate question pairs from triples using vLLM, then compute question embeddings."
    )
    parser.add_argument("--benchmark", "-b", type=str, required=True,
                        help="Benchmark name, e.g., 'hotpotqa'. Expects data/{benchmark}/{benchmark}_corpus_triple.json")
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen2.5-32B-Instruct",
                        help="Model name for vLLM question generation (default: Qwen/Qwen2.5-32B-Instruct).")
    parser.add_argument("--embedding-model", "-e", type=str, default="nvidia/NV-Embed-v2",
                        help="Embedding model name (default: nvidia/NV-Embed-v2). Must be supported by embedding_model.py.")
    parser.add_argument("--device", "-d", type=str, default="cuda:0",
                        help="Device for embedding model (default: cuda:0).")
    args = parser.parse_args()

    # Build paths
    base_dir = "data"
    input_triple = f"{base_dir}/{args.benchmark}/{args.benchmark}_corpus_triple.json"
    qa_output = f"{base_dir}/{args.benchmark}/{args.benchmark}_questions.jsonl"
    embedding_output = f"{base_dir}/{args.benchmark}/{args.benchmark}_questions_embeddings.parquet"

    # Skip if the final Parquet already exists
    if os.path.exists(embedding_output):
        print(f"Output Parquet already exists: {embedding_output}")
        print("Skipping question generation and embedding computation.")
        return

    print(f"Benchmark: {args.benchmark}")
    print(f"vLLM model: {args.model}")
    print(f"Embedding model: {args.embedding_model} on {args.device}")
    print(f"Input triple file: {input_triple}")
    print(f"question output: {qa_output}")
    print(f"Embedding output: {embedding_output}")

    # Step 1: Generate question pairs using vLLM
    generate_qa_pairs(input_triple, qa_output, args.model)

    # Step 2: Compute embeddings for questions
    compute_embeddings(qa_output, embedding_output, args.embedding_model, args.device)

if __name__ == "__main__":
    main()
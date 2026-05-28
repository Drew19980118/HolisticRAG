import json
import time
import argparse
import os
import pandas as pd
import requests
from typing import List, Dict, Optional
from prompts.seed_query_extraction import SEED_QUERY_PROMPT_TEMPLATE
from embedding_model import get_embedding_model

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


def extract_questions(json_file_path: str) -> List[str]:
    """
    Read a JSON file containing a list of dictionaries,
    extract the 'question' field from each dictionary.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Root element of JSON file must be a list")

    questions = []
    for item in data:
        if isinstance(item, dict) and 'question' in item:
            questions.append(item['question'])
        else:
            print(f"Warning: Skipping invalid item {item}")

    return questions


def generate_seed_queries(query: str, model_name: str) -> List[str]:
    """Generate seed queries using the imported prompt template."""
    messages = [{"role": "user", "content": SEED_QUERY_PROMPT_TEMPLATE.format(query=query)}]
    response = query_vllm_messages(messages, model_name=model_name)

    if response is None:
        return []

    # Parse expected format: "Seed queries: query1; query2; query3"
    seed_queries = []
    marker = "Seed queries:"
    idx = response.find(marker)
    if idx != -1:
        after_marker = response[idx + len(marker):].strip()
        parts = after_marker.split(';')
        for part in parts:
            part = part.strip()
            if part:
                seed_queries.append(part)
    else:
        # Fallback: treat entire response as a single candidate
        print(f"Warning: 'Seed queries:' not found in response for query: {query}")
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line:
                seed_queries.append(line)

    return seed_queries


def run_query_seed_extraction(benchmark: str, model_name: str) -> str:
    """
    Extract seed queries for the given benchmark and save them to a JSON file.
    Returns the path to the saved JSON file.
    """
    base_dir = "data"
    input_file = os.path.join(base_dir, f"{benchmark}.json")
    output_dir = os.path.join(base_dir, benchmark, "seed_queries")
    output_file = os.path.join(output_dir, f"{benchmark}_seed_queries.json")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Extracting seed queries for benchmark: {benchmark}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    queries = extract_questions(input_file)
    results = []
    total = len(queries)
    for idx, q in enumerate(queries):
        print(f"Processing [{idx + 1}/{total}]: {q[:80]}...")
        seed_qs = generate_seed_queries(q, model_name=model_name)
        results.append({
            "query": q,
            "seed_queries": seed_qs
        })
        time.sleep(0.5)  # avoid overwhelming the API

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Seed extraction completed. Saved to {output_file}")
    return output_file


def compute_embeddings(seed_json_path: str, benchmark: str, embedding_model_name: str, device: str = "cuda:0"):
    """
    Load seed queries JSON, compute embeddings for seed queries and parent queries,
    and save to a Parquet file.
    """
    print(f"Loading seed queries from {seed_json_path}...")
    with open(seed_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Seed queries file must contain a JSON array")

    # Collect all seed queries and their parent queries
    all_seed_queries = []
    all_parent_queries = []

    for item in data:
        original_query = item.get("query")
        seed_queries = item.get("seed_queries", [])
        if not original_query or not seed_queries:
            continue
        for sq in seed_queries:
            if sq:
                all_seed_queries.append(sq)
                all_parent_queries.append(original_query)

    print(f"Collected {len(all_seed_queries)} seed queries")

    if not all_seed_queries:
        raise ValueError("No valid seed queries found")

    # Initialize embedding model using decoupled class
    embed_model = get_embedding_model(embedding_model_name=embedding_model_name, device=device)

    print("Computing embeddings for seed queries...")
    seed_embeddings = embed_model.batch_encode(all_seed_queries)

    # Unique parent queries and their embeddings
    unique_queries = list(set(all_parent_queries))
    print(f"Computing embeddings for {len(unique_queries)} unique parent queries...")
    query_embeddings_array = embed_model.batch_encode(unique_queries)
    query_to_embedding = {q: emb for q, emb in zip(unique_queries, query_embeddings_array)}

    parent_query_embeddings = [query_to_embedding[q] for q in all_parent_queries]

    # Build DataFrame
    df = pd.DataFrame({
        "query": all_parent_queries,
        "seed_query_text": all_seed_queries,
        "seed_query_embedding": list(seed_embeddings),
        "query_embedding": parent_query_embeddings
    })

    # Save to Parquet
    output_dir = os.path.join("data", benchmark, "seed_queries")
    os.makedirs(output_dir, exist_ok=True)
    output_parquet = os.path.join(output_dir, f"{benchmark}_seed_queries_embeddings.parquet")
    print(f"Saving to {output_parquet}...")
    df.to_parquet(output_parquet, index=False)
    print("Embedding computation completed.")


# ==================== Main pipeline ====================
def main():
    parser = argparse.ArgumentParser(
        description="Extract seed queries from a benchmark dataset using vLLM, "
                    "then compute embeddings with a specified embedding model."
    )
    parser.add_argument("--benchmark", "-b", type=str, required=True,
                        help="Benchmark name (e.g., 'musique'). Expects data/{benchmark}.json as input.")
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen2.5-32B-Instruct",
                        help="Model name for vLLM seed extraction (default: Qwen/Qwen2.5-32B-Instruct).")
    parser.add_argument("--embedding-model", "-e", type=str, default="nvidia/NV-Embed-v2",
                        help="Embedding model name (default: nvidia/NV-Embed-v2). Must support AutoModel and .encode().")
    parser.add_argument("--device", "-d", type=str, default="cuda:0",
                        help="Device for embedding model (default: cuda:0).")
    args = parser.parse_args()

    # Check if the final output Parquet already exists
    output_parquet = os.path.join("data", args.benchmark, "seed_queries",
                                  f"{args.benchmark}_seed_queries_embeddings.parquet")
    if os.path.exists(output_parquet):
        print(f"Output Parquet already exists: {output_parquet}")
        print("Skipping seed extraction and embedding computation.")
        return

    # Step 1: seed extraction (always performed if Parquet missing)
    seed_json_path = run_query_seed_extraction(args.benchmark, args.model)

    # Step 2: compute embeddings with decoupled model
    compute_embeddings(seed_json_path, args.benchmark, args.embedding_model, device=args.device)

if __name__ == "__main__":
    main()
import json
import argparse
import os
import pandas as pd
from embedding_model import get_embedding_model   # decoupled embedding factory


def main():
    parser = argparse.ArgumentParser(
        description="Compute embeddings for passage texts in a JSONL corpus."
    )
    parser.add_argument("--benchmark", "-b", type=str, required=True,
                        help="Benchmark name, e.g., 'hotpotqa'. Will read "
                             "data/{benchmark}/{benchmark}_corpus_triple.json and "
                             "write data/{benchmark}/{benchmark}_chunks_embedding.parquet")
    parser.add_argument("--embedding-model", "-e", type=str, default="nvidia/NV-Embed-v2",
                        help="Embedding model name (default: nvidia/NV-Embed-v2). "
                             "Supported: NV-Embed-v2, e5-mistral")
    parser.add_argument("--device", "-d", type=str, default="cuda:0",
                        help="Device for embedding model (default: cuda:0)")
    args = parser.parse_args()

    # Build input/output paths from benchmark name
    base_dir = "data"
    input_file = os.path.join(base_dir, args.benchmark, f"{args.benchmark}_corpus_triple.json")
    output_file = os.path.join(base_dir, args.benchmark, f"{args.benchmark}_chunks_embeddings.parquet")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Skip if output already exists
    if os.path.exists(output_file):
        print(f"Output Parquet already exists: {output_file}")
        print("Skipping embedding computation.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Benchmark: {args.benchmark}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    # Initialize embedding model using factory
    print("Initializing embedding model...")
    embed_model = get_embedding_model(args.embedding_model, device=args.device)

    # Read JSONL file: each line is a JSON object with "idx" and "text"
    records = []      # list of dicts with idx and text
    all_texts = []    # list of passage texts for batch encoding

    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: line {line_num} JSON decode error, skipped - {e}")
                continue

            idx = data.get("idx")
            text = data.get("text")
            if idx is None or text is None:
                print(f"Warning: line {line_num} missing 'idx' or 'text', skipped")
                continue
            if not text.strip():
                print(f"Warning: line {line_num} empty text, skipped")
                continue

            records.append({"idx": idx, "text": text})
            all_texts.append(text)

    print(f"Collected {len(records)} valid passages")

    if not records:
        print("No valid data, exiting.")
        return

    # Compute embeddings in batches
    print("Computing embeddings...")
    embeddings = embed_model.batch_encode(all_texts)   # shape (n, dim)

    # Build DataFrame and save as Parquet
    df = pd.DataFrame(records)
    df["passage_embedding"] = list(embeddings)

    print(f"Saving to {output_file}...")
    df.to_parquet(output_file, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
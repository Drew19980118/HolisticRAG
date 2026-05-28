import json
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import requests
import time
import re

try:
    import hdbscan
    import umap
except ImportError as e:
    print("Error: hdbscan and umap-learn libraries are required.")
    print("Please run: pip install hdbscan umap-learn")
    raise e

# Import decoupled embedding model factory
from embedding_model import get_embedding_model

# Import prompt templates
from prompts.filter import FILTER_PROMPT_TEMPLATE
from prompts.sufficiency import SUFFICIENCY_PROMPT_TEMPLATE
from prompts.generation import GENERATION_PROMPT_TEMPLATE
from prompts.fallback import FALLBACK_PROMPT_TEMPLATE


# ================== Vector utilities ==================
def load_embeddings(df: pd.DataFrame, embedding_col: str) -> np.ndarray:
    """Convert the embedding column (list or numpy array) in DataFrame to a 2D numpy array"""
    emb_list = df[embedding_col].tolist()
    return np.array([np.array(e) for e in emb_list])


def min_max_normalize(x: np.ndarray) -> np.ndarray:
    """Min-max normalization, input is a 1D array"""
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    if range_val == 0:
        return np.ones_like(x)
    return (x - min_val) / range_val


def find_most_similar(query_emb: np.ndarray, target_embeddings: np.ndarray,
                      target_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], float]:
    """Use dot product + min-max normalization to find the most similar record. Returns (text, triple, normalized_score)"""
    if len(target_embeddings) == 0:
        return None, None, 0.0
    scores = np.dot(target_embeddings, query_emb)
    normalized_scores = min_max_normalize(scores)
    best_idx = np.argmax(normalized_scores)
    best_score = normalized_scores[best_idx]
    row = target_df.iloc[best_idx]
    return row['passage_text'], row['triple'], best_score


def find_top_k_similar(query: str,
                       target_embeddings: np.ndarray,
                       target_df: pd.DataFrame,
                       get_embedding_func,
                       k: int = 5) -> List[Tuple[str, float, int]]:
    """
    Returns list of (text, normalized_score, integer_position_index)
    """
    if len(target_embeddings) == 0:
        return []

    query_embedding = get_embedding_func(query)
    if query_embedding.ndim == 2:
        query_embedding = query_embedding.squeeze()

    scores = np.dot(target_embeddings, query_embedding)
    if scores.ndim == 2:
        scores = scores.squeeze()

    normalized_scores = min_max_normalize(scores)
    sorted_indices = np.argsort(normalized_scores)[::-1]
    top_k_indices = sorted_indices[:k]

    results = []
    for pos in top_k_indices:
        row = target_df.iloc[pos]
        text = row['content']
        score = normalized_scores[pos]
        results.append((text, score, pos))
    return results


def retrieve_top_k_passages(query: str,
                            passage_embeddings: np.ndarray,
                            passage_df: pd.DataFrame,
                            get_embedding_func,
                            k: int = 5) -> List[Tuple[str, float, int]]:
    """
    Retrieve top-k similar passages from the passage corpus.
    Returns list of (passage_text, normalized_score, idx)
    """
    if len(passage_embeddings) == 0:
        return []

    query_embedding = get_embedding_func(query)
    if query_embedding.ndim == 2:
        query_embedding = query_embedding.squeeze()

    scores = np.dot(passage_embeddings, query_embedding)
    if scores.ndim == 2:
        scores = scores.squeeze()

    normalized_scores = min_max_normalize(scores)
    sorted_indices = np.argsort(normalized_scores)[::-1]
    top_k_indices = sorted_indices[:k]

    results = []
    for pos in top_k_indices:
        row = passage_df.iloc[pos]
        text = row['text']
        score = normalized_scores[pos]
        idx_val = row.get('idx', pos)
        results.append((text, score, idx_val))
    return results


# ================== HDBSCAN clustering + LLM filtering ==================
def cluster_and_filter_questions(
    seed_query: str,
    seed_emb: np.ndarray,
    top_k_results: List[Tuple[str, float, int]],
    qa_df: pd.DataFrame,
    llm_filter_func,
    min_cluster_size: int = 2,
) -> List[Dict]:
    """
    Perform HDBSCAN clustering on the candidate questions, take the cluster containing the top-1 result,
    and pass the other questions in that cluster to an LLM filter.
    """
    if not top_k_results:
        return []

    candidate_texts = [res[0] for res in top_k_results]
    candidate_indices = [res[2] for res in top_k_results]

    embeddings_list = []
    for idx in candidate_indices:
        row = qa_df.iloc[idx]
        emb = row['embedding']
        if isinstance(emb, list):
            emb = np.array(emb)
        embeddings_list.append(emb)
    X = np.vstack(embeddings_list)
    n_samples = X.shape[0]

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    if n_samples <= 1:
        selected_questions = candidate_texts
        print(f"  [Cluster] Only {n_samples} candidate(s), using directly")
    else:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='leaf'
        )
        labels = clusterer.fit_predict(X)

        top1_idx = 0
        top1_label = labels[top1_idx]
        top1_question = candidate_texts[top1_idx]

        if top1_label == -1:
            selected_questions = [top1_question]
            print(f"  [Cluster] Top-1 is noise, keeping only itself")
        else:
            cluster_indices = [i for i, lab in enumerate(labels) if lab == top1_label]
            cluster_questions = [candidate_texts[i] for i in cluster_indices]
            print(f"  [Cluster] Cluster {top1_label} contains {len(cluster_questions)} questions")

            other_questions = [q for q in cluster_questions if q != top1_question]
            selected_questions = [top1_question]

            if other_questions:
                filtered_others = llm_filter_func(seed_query, other_questions)
                if filtered_others:
                    selected_questions.extend(filtered_others)
                    print(f"  [LLM Filter] Kept {len(filtered_others)} out of {len(other_questions)} other questions")
                else:
                    print(f"  [LLM Filter] No other questions kept, using only top-1")
            else:
                print(f"  [Cluster] Cluster contains only top-1, no other questions")

    result_map = {}
    for q_text in selected_questions:
        matching_rows = qa_df[qa_df['content'] == q_text]
        for _, row in matching_rows.iterrows():
            key = (row['triple'], row['passage_text'])
            if key not in result_map:
                result_map[key] = {
                    'text': q_text,
                    'triple': row['triple'],
                    'passage_text': row['passage_text']
                }
    return list(result_map.values())


def llm_filter_questions(seed_query: str, candidate_questions: List[str],
                         llm_model_name: str) -> List[str]:
    """
    Call LLM to filter candidate_questions, returning only those highly relevant to the seed query.
    """
    if not candidate_questions:
        return []

    numbered_list = "\n".join(f"{i+1}. {q}" for i, q in enumerate(candidate_questions))
    prompt = FILTER_PROMPT_TEMPLATE.format(seed_query=seed_query, numbered_list=numbered_list)

    messages = [{"role": "user", "content": prompt}]
    llm_output = query_vllm_messages(messages, model_name=llm_model_name)
    if llm_output is None:
        print("  [LLM Filter] Call failed, returning original list")
        return candidate_questions

    try:
        filtered_indices = json.loads(llm_output)
        if isinstance(filtered_indices, list) and all(isinstance(i, int) for i in filtered_indices):
            selected = []
            seen = set()
            for idx in filtered_indices:
                zero_idx = idx - 1
                if 0 <= zero_idx < len(candidate_questions) and zero_idx not in seen:
                    seen.add(zero_idx)
                    selected.append(candidate_questions[zero_idx])
            if selected:
                return selected
            else:
                print("  [LLM Filter] Parsed indices are invalid, returning original list")
                return candidate_questions
        else:
            print(f"  [LLM Filter] Output is not an integer list: {llm_output}, returning original list")
            return candidate_questions
    except json.JSONDecodeError:
        numbers = re.findall(r'\b\d+\b', llm_output)
        if numbers:
            selected = []
            seen = set()
            for num_str in numbers:
                idx = int(num_str) - 1
                if 0 <= idx < len(candidate_questions) and idx not in seen:
                    seen.add(idx)
                    selected.append(candidate_questions[idx])
            if selected:
                return selected
        print(f"  [LLM Filter] Could not parse output: {llm_output}, returning original list")
        return candidate_questions


# ================== LLM call ==================
def query_vllm_messages(messages: List[Dict[str, str]], model_name: str,
                        max_retries: int = 3) -> Optional[str]:
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


# ========== Prompt building functions (using imported templates) ==========
def build_sufficiency_prompt(original_query: str, passages: List[str], triples: List[str]) -> str:
    unique_passages = list(dict.fromkeys(passages))
    unique_triples = list(dict.fromkeys(triples))
    context = ""
    if unique_passages:
        context += "[Known Passages]:\n"
        for i, p in enumerate(unique_passages, 1):
            context += f"{i}. {p}\n"
    if unique_triples:
        context += "[Known Triples]:\n"
        for i, t in enumerate(unique_triples, 1):
            context += f"{i}. {t}\n"
    return SUFFICIENCY_PROMPT_TEMPLATE.format(original_query=original_query, context=context)


def build_query_generation_prompt(original_query: str, passages: List[str], triples: List[str],
                                  previous_queries: List[str]) -> str:
    unique_passages = list(dict.fromkeys(passages))
    unique_triples = list(dict.fromkeys(triples))
    context = ""
    if unique_passages:
        context += "[Known Passages]:\n"
        for i, p in enumerate(unique_passages, 1):
            context += f"{i}. {p}\n"
    if unique_triples:
        context += "[Known Triples]:\n"
        for i, t in enumerate(unique_triples, 1):
            context += f"{i}. {t}\n"
    # previous_queries is not used in the current template but kept for potential future use
    return GENERATION_PROMPT_TEMPLATE.format(original_query=original_query, context=context)


def build_fallback_prompt(original_query: str, top_passages: List[str]) -> str:
    context = "[Retrieved Passages]:\n"
    for i, p in enumerate(top_passages, 1):
        context += f"{i}. {p}\n"
    return FALLBACK_PROMPT_TEMPLATE.format(context=context, original_query=original_query)


def parse_sufficiency_output(output: str) -> Tuple[str, Optional[str]]:
    output = output.strip()
    if output.startswith("Yes."):
        match = re.search(r"Yes\.\s*Answer:\s*(.*)", output, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            return "Yes", answer
        rest = output[4:].strip()
        if rest.startswith("Answer:"):
            rest = rest[7:].strip()
        return "Yes", rest
    elif output.startswith("No."):
        return "No", None
    else:
        print(f"Warning: Could not parse sufficiency output: {output}, treating as 'No'")
        return "No", None


def parse_query_generation_output(output: str) -> Optional[str]:
    output = output.strip()
    match = re.search(r"Query:\s*(.*)", output, re.DOTALL)
    if match:
        query = match.group(1).strip()
        if query:
            return query
    match_old = re.search(r"No\.\s*Query:\s*(.*)", output, re.DOTALL)
    if match_old:
        query = match_old.group(1).strip()
        if query:
            return query
    print(f"Warning: Could not parse query generation output: {output}")
    return None


# ================== Main pipeline ==================
def process_one_query(original_query: str,
                      qa_df: pd.DataFrame,
                      qa_embeddings: np.ndarray,
                      seed_df: pd.DataFrame,
                      chunk_df: pd.DataFrame,
                      chunk_embeddings: np.ndarray,
                      text_to_idx_list: Dict[str, List[int]],
                      get_embedding_func,
                      llm_model_name: str,
                      max_iter: int = 5,
                      repeat_sim_threshold: float = 0.90,
                      topk_retrieval: int = 30) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Process a single original query using seed queries, clustering, and iterative retrieval.
    """
    seed_rows = seed_df[seed_df['query'] == original_query]
    if seed_rows.empty:
        print(f"Warning: No seed queries found for original query '{original_query}'")
        return None, {"status": "error", "reason": "no_seed_queries"}

    seed_queries_list = []
    for _, row in seed_rows.iterrows():
        if 'seed_query_text' in row and row['seed_query_text']:
            seed_queries_list.append(row['seed_query_text'])
        else:
            seed_queries_list.append(f"seed_{_}")

    all_initial_evidences = []
    for _, row in seed_rows.iterrows():
        seed_q_text = row['seed_query_text']
        seed_emb = row['seed_query_embedding']
        if isinstance(seed_emb, list):
            seed_emb = np.array(seed_emb)

        print(f"\nProcessing seed query: {seed_q_text}")

        top_results = find_top_k_similar(
            seed_q_text,
            qa_embeddings,
            qa_df,
            get_embedding_func,
            k=topk_retrieval
        )
        if not top_results:
            print(f"  Warning: No matches retrieved for seed query")
            continue

        # Closure to pass llm_model_name into the filter
        def filter_with_model(seed_q, candidates):
            return llm_filter_questions(seed_q, candidates, llm_model_name)

        filtered_evidences = cluster_and_filter_questions(
            seed_query=seed_q_text,
            seed_emb=seed_emb,
            top_k_results=top_results,
            qa_df=qa_df,
            llm_filter_func=filter_with_model
        )
        all_initial_evidences.extend(filtered_evidences)
        print(f"  Obtained {len(filtered_evidences)} valid evidences from this seed query")

    unique_evidences = {}
    for ev in all_initial_evidences:
        key = (ev['triple'], ev['passage_text'])
        if key not in unique_evidences:
            unique_evidences[key] = ev
    initial_evidences = list(unique_evidences.values())
    print(f"\nTotal unique evidences collected: {len(initial_evidences)} (triple+passage)")

    passages = [ev['passage_text'] for ev in initial_evidences]
    triples = [ev['triple'] for ev in initial_evidences]
    passage_idxs = []
    for ev in initial_evidences:
        p_text = ev['passage_text']
        idx_list = text_to_idx_list.get(p_text, [])
        passage_idxs.append(idx_list)

    previous_queries = list(seed_queries_list)
    intermediate_steps = []
    last_generated_emb = None

    for round_num in range(1, max_iter + 1):
        print(f"\n=== Round {round_num} ===")
        print(f"Current known passages: {len(passages)}, triples: {len(triples)}")
        if previous_queries:
            print(f"Previously Asked Questions (including seed queries): {previous_queries}")

        sufficiency_prompt = build_sufficiency_prompt(original_query, passages, triples)
        messages = [{"role": "user", "content": sufficiency_prompt}]
        llm_output = query_vllm_messages(messages, model_name=llm_model_name)
        if llm_output is None:
            print("LLM call failed, terminating")
            return None, {"status": "error", "reason": "llm_failure"}

        print(f"[Sufficiency] LLM output: {llm_output}")
        decision, answer = parse_sufficiency_output(llm_output)

        if decision == "Yes":
            print(f"Successfully obtained answer: {answer}")
            run_info = {
                "status": "success",
                "original_query": original_query,
                "seed_queries": seed_queries_list,
                "intermediate_steps": intermediate_steps,
                "final_answer": answer,
                "total_rounds": round_num,
                "all_passage_idxs": passage_idxs
            }
            return answer, run_info

        print("Information insufficient, need to generate a new query...")
        gen_prompt = build_query_generation_prompt(original_query, passages, triples, previous_queries)
        messages = [{"role": "user", "content": gen_prompt}]
        llm_output = query_vllm_messages(messages, model_name=llm_model_name)
        if llm_output is None:
            print("LLM call failed during new query generation, terminating")
            return None, {"status": "error", "reason": "llm_failure_gen"}

        print(f"[Generation] LLM output: {llm_output}")
        new_query = parse_query_generation_output(llm_output)
        if not new_query:
            print("Failed to extract new query, terminating")
            return None, {"status": "error", "reason": "empty_new_query"}

        new_emb = get_embedding_func(new_query)
        if new_emb is None:
            print("Failed to generate embedding for new query, terminating")
            return None, {"status": "error", "reason": "embedding_failure"}
        if isinstance(new_emb, list):
            new_emb = np.array(new_emb)

        if last_generated_emb is not None:
            sim_with_prev = np.dot(new_emb, last_generated_emb)
            print(f"Cosine similarity between new query and previous generated query: {sim_with_prev:.4f}")
            if sim_with_prev >= repeat_sim_threshold:
                print(f"Similarity {sim_with_prev:.4f} exceeds threshold {repeat_sim_threshold}, entering fallback early")
                step_info = {
                    "round": round_num,
                    "generated_query": new_query,
                    "retrieved_passage": None,
                    "retrieved_passage_idxs": [],
                    "retrieved_triple": None,
                    "similarity": 0.0,
                    "repeat_detected": True,
                    "similarity_with_prev": float(sim_with_prev)
                }
                intermediate_steps.append(step_info)
                break
        last_generated_emb = new_emb

        previous_queries.append(new_query)
        print(f"Need to further query: {new_query}")

        text, triple, sim = find_most_similar(new_emb, qa_embeddings, qa_df)
        if text and triple:
            idx_list = text_to_idx_list.get(text, [])
            print(f"Retrieved new passage (similarity {sim:.4f}) idxs={idx_list}: {text[:100]}...")
            step_info = {
                "round": round_num,
                "generated_query": new_query,
                "retrieved_passage": text,
                "retrieved_passage_idxs": idx_list,
                "retrieved_triple": triple,
                "similarity": float(sim),
                "repeat_detected": False
            }
            intermediate_steps.append(step_info)
            passages.append(text)
            triples.append(triple)
            passage_idxs.append(idx_list)
        else:
            print("No similar result found, cannot continue")
            step_info = {
                "round": round_num,
                "generated_query": new_query,
                "retrieved_passage": None,
                "retrieved_passage_idxs": [],
                "retrieved_triple": None,
                "similarity": 0.0,
                "repeat_detected": False
            }
            intermediate_steps.append(step_info)
            break

    print(f"\nReached maximum iterations, retrieval failure, or repeat generation. Falling back to retrieve top-5 passages with original query and generate final answer.")
    if 'query_embedding' in seed_df.columns:
        original_row = seed_df[seed_df['query'] == original_query].iloc[0]
        query_emb = original_row['query_embedding']
        if isinstance(query_emb, list):
            query_emb = np.array(query_emb)
    else:
        print("Warning: 'query_embedding' not found in seed_queries_embeddings.parquet, computing on the fly")
        query_emb = get_embedding_func(original_query)
        if query_emb is None:
            return None, {"status": "error", "reason": "fallback_embedding_failure"}
        if isinstance(query_emb, list):
            query_emb = np.array(query_emb)

    top_passages_info = retrieve_top_k_passages(
        original_query,
        chunk_embeddings,
        chunk_df,
        get_embedding_func,
        k=5
    )
    top_passages = [p for p, _, _ in top_passages_info]
    top_passages_idxs = [text_to_idx_list.get(p, []) for p in top_passages]

    if not top_passages:
        print("No passages retrieved, cannot answer")
        return None, {"status": "error", "reason": "no_passages_retrieved"}

    fallback_prompt = build_fallback_prompt(original_query, top_passages)
    messages = [{"role": "user", "content": fallback_prompt}]
    llm_output = query_vllm_messages(messages, model_name=llm_model_name)
    if llm_output is None:
        print("Fallback LLM call failed")
        return None, {"status": "error", "reason": "fallback_llm_failure"}

    if llm_output.startswith("Answer:"):
        final_answer = llm_output[7:].strip()
    else:
        final_answer = llm_output.strip()

    print(f"Fallback answer: {final_answer}")

    run_info = {
        "status": "fallback",
        "original_query": original_query,
        "seed_queries": seed_queries_list,
        "retrieved_passages": top_passages,
        "retrieved_passages_idxs": top_passages_idxs,
        "final_answer": final_answer,
        "total_rounds": len(intermediate_steps),
        "fallback_reason": "repeat_detected" if (intermediate_steps and intermediate_steps[-1].get("repeat_detected")) else "max_iter_or_no_result"
    }
    return final_answer, run_info


def run_pipeline(benchmark: str,
                 embedding_model_name: str,
                 llm_model_name: str,
                 device: str = "cuda:1",
                 max_iter: int = 5,
                 repeat_sim_threshold: float = 0.90) -> List[Dict[str, Any]]:
    """
    Run the pipeline for the given benchmark using the specified embedding and LLM models.
    Data paths are built as: data/{benchmark}/{benchmark}_*.parquet
    """
    # Build file paths
    base_data_dir = "data"
    qa_pairs_path = f"{base_data_dir}/{benchmark}/{benchmark}_questions_embeddings.parquet"
    seed_queries_path = f"{base_data_dir}/{benchmark}/{benchmark}_seed_queries_embeddings.parquet"
    chunk_path = f"{base_data_dir}/{benchmark}/{benchmark}_chunks_embeddings.parquet"
    output_jsonl = f"{base_data_dir}/{benchmark}/{benchmark}_results.jsonl"

    print(f"Benchmark: {benchmark}")
    print(f"Embedding model: {embedding_model_name} on {device}")
    print(f"LLM model: {llm_model_name}")
    print(f"Questions file: {qa_pairs_path}")
    print(f"Seed queries file: {seed_queries_path}")
    print(f"Passage corpus file: {chunk_path}")
    print(f"Output file: {output_jsonl}")

    # Initialize embedding model using the factory
    print("Initializing embedding model...")
    embed_model = get_embedding_model(embedding_model_name, device=device)

    def get_embedding(text: str) -> np.ndarray:
        return embed_model.encode_single(text)

    # Load data
    qa_df = pd.read_parquet(qa_pairs_path)
    if 'embedding' not in qa_df.columns:
        raise ValueError("questions_embeddings.parquet missing 'embedding' column")
    qa_embeddings = load_embeddings(qa_df, 'embedding')
    print(f"Loaded QA pairs: {len(qa_df)}")

    seed_df = pd.read_parquet(seed_queries_path)
    if 'query' not in seed_df.columns or 'seed_query_embedding' not in seed_df.columns:
        raise ValueError("seed_queries_embeddings.parquet missing 'query' or 'seed_query_embedding' column")
    if 'query_embedding' not in seed_df.columns:
        print("Warning: 'query_embedding' not found in seed_queries_embeddings.parquet, will compute on-the-fly during fallback")
    print(f"Loaded Seed queries: {len(seed_df)}")

    chunk_df = pd.read_parquet(chunk_path)
    if 'passage_embedding' not in chunk_df.columns:
        raise ValueError("chunks_embeddings.parquet missing 'passage_embedding' column")
    if 'text' not in chunk_df.columns:
        raise ValueError("chunks_embeddings.parquet missing 'text' column")
    if 'idx' not in chunk_df.columns:
        print("Warning: chunks_embeddings.parquet has no 'idx' column, using row index as idx")
        chunk_df['idx'] = chunk_df.index
    chunk_embeddings = load_embeddings(chunk_df, 'passage_embedding')

    text_to_idx_list = {}
    for _, row in chunk_df.iterrows():
        text = row['text']
        idx_val = row['idx']
        if text not in text_to_idx_list:
            text_to_idx_list[text] = []
        text_to_idx_list[text].append(idx_val)

    print(f"Loaded passages: {len(chunk_df)} items, unique text count: {len(text_to_idx_list)}")

    # Get unique original queries from seed_df
    test_queries = seed_df['query'].unique().tolist()
    print(f"Found {len(test_queries)} unique original queries to process")

    all_results = []
    total_queries = len(test_queries)
    success_count = 0
    error_count = 0
    fallback_count = 0

    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for idx, q in enumerate(test_queries, 1):
            print(f"\n========== Processing {idx}/{total_queries}: {q} ==========")
            answer, run_info = process_one_query(
                q, qa_df, qa_embeddings, seed_df,
                chunk_df, chunk_embeddings, text_to_idx_list,
                get_embedding, llm_model_name,
                max_iter=max_iter,
                repeat_sim_threshold=repeat_sim_threshold
            )
            run_info["query_index"] = idx - 1
            all_results.append(run_info)
            fout.write(json.dumps(run_info, ensure_ascii=False) + "\n")
            fout.flush()

            status = run_info.get("status")
            if status == "success":
                success_count += 1
            elif status == "fallback":
                fallback_count += 1
            else:
                error_count += 1

            print(f"Progress: {idx}/{total_queries} completed. Success: {success_count}, fallback: {fallback_count}, error: {error_count}")

    print(f"\nProcessing complete. Total: {len(all_results)}, success: {success_count}, fallback: {fallback_count}, error: {error_count}")
    return all_results


# ================== Main entry point ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run QA pipeline with seed queries, question embeddings, and passage corpus."
    )
    parser.add_argument("--benchmark", "-b", type=str, required=True,
                        help="Benchmark name (e.g., 'hotpotqa'). Data files are expected in data/{benchmark}/")
    parser.add_argument("--embedding-model", "-e", type=str, default="nvidia/NV-Embed-v2",
                        help="Embedding model name (default: nvidia/NV-Embed-v2)")
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen2.5-32B-Instruct",
                        help="LLM model name for vLLM (default: Qwen/Qwen2.5-32B-Instruct)")
    parser.add_argument("--device", "-d", type=str, default="cuda:0",
                        help="Device for embedding model (default: cuda:0)")
    parser.add_argument("--max-iter", type=int, default=5,
                        help="Maximum number of iteration rounds (default: 5)")
    parser.add_argument("--repeat-sim-threshold", type=float, default=0.90,
                        help="Cosine similarity threshold for early fallback (default: 0.90)")
    args = parser.parse_args()

    run_pipeline(
        benchmark=args.benchmark,
        embedding_model_name=args.embedding_model,
        llm_model_name=args.model,
        device=args.device,
        max_iter=args.max_iter,
        repeat_sim_threshold=args.repeat_sim_threshold
    )
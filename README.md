<div align= "center">
    <h1> Towards Holistic Knowledge Retrieval for Augmented Generation: Integrating Graph-Grounded and Query-Aligned Knowledge </h1>
</div>

## Overview

![](./assets/overview.png)

## Abstract

Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to access external knowledge during generation, improving factual reliability without parameter retraining. However, existing methods still struggle to retrieve evidence that is both relevant and complete. Standard RAG may miss semantic associations among related chunks. Graph-based RAG addresses this issue by modeling entity relations, but it is often query-agnostic. Question-format RAG further improves query matching by converting facts into Question-Answer (QA) pairs, but it cannot guarantee semantic association and introduces knowledge fragmentation. To address these issues, we propose HolisticRAG, a framework integrating graph‑grounded and query-aligned knowledge for holistic knowledge RAG. In the offline stage, HolisticRAG constructs a knowledge representation to preserve semantic associations while reducing query-agnostic retrieval. In the online stage, it applies holistic knowledge clustering to merge all fragmented evidence and adaptively retrieve sufficient supporting facts. Experiments demonstrate the effiectiveness of HolisticRAG.

## Installation

We recommend using Conda to manage the environment. All experiments are conducted with Python 3.10.

```sh
conda create -n HolisticRAG python=3.10
conda activate HolisticRAG
pip install -r requirements.txt

## Offline Stage

The offline stage constructs a knowledge representation that preserves semantic associations while reducing query‑agnostic retrieval. It consists of three sequential steps.

### 1. Fine-grained Fact Extraction

We first extract fine‑grained atomic facts from each document in the benchmark. Each fact is a short, self‑contained textual statement.

```sh
python fact_extraction.py --benchmark hotpotqa --model Qwen/Qwen2.5-32B-Instruct

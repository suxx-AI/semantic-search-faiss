# Semantic Search Engine with FAISS

A lightweight, high-speed semantic search pipeline built using Hugging Face `transformers` and Meta's `FAISS` (Facebook AI Similarity Search) backend. This project showcases how to implement the **Retrieval** layer of a Retrieval-Augmented Generation (RAG) system using localized dataset streaming.

This is basically the entire **Retrieval** layer used in modern RAG pipelines.

## How it works
1. **Streaming Data:** Instead of downloading the full `dair-ai/emotion` dataset to disk and killing my RAM, the code streams it directly from Hugging Face cloud in optimized batches of 32.
2. **Generating Embeddings:** It passes the text through the `multi-qa-mpnet-base-dot-v1` model, extracts the `[CLS]` token, and turns every tweet into a 768-dimensional vector coordinate.
3. **FAISS Indexing:** It loads all those coordinates into a local FAISS index. FAISS clusters the vector space so it can skip 99% of the data and find matching vectors instantly.
4. **Search Engine:** When you pass a query like `"im feeling playful already"`, it converts it to a vector, calculates the L2/Euclidean distance, and pulls the top $k$ nearest matches in milliseconds.

## The Code Stack
* Python
* PyTorch (CUDA/GPU accelerated)
* Hugging Face (`transformers` & `datasets`)
* FAISS (C++ vector engine under the hood)
# SciRAG-Eval

Evaluation-first Retrieval-Augmented Generation (RAG) system over scientific and dental PDFs.

## Goals

- Ingest and chunk scientific pdfs.
- Build a vector search index over document chunks
- Answer questions with a RAG pipeline using LLMs
- Provide citations to exact source locations (paper + page)
- Include a custom evaluation harness to measure accuracy and hallucinations
- Offer a Streamlit UI for interactive QA and error analysis

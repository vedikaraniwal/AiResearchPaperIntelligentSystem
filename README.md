# AI Research Paper Intelligent System

This project builds an intelligent workflow for exploring and analyzing AI research papers using modern NLP and vector search techniques. It loads a dataset of arXiv papers, creates text embeddings, indexes them with FAISS, and enables semantic search over paper titles and abstracts.

## What this project does

- Downloads and prepares a paper dataset from Hugging Face
- Converts paper titles and abstracts into embeddings using Sentence Transformers
- Stores and indexes embeddings using FAISS for fast similarity search
- Searches for relevant papers using semantic similarity
- Generates:
  - paper summaries
  - keyword extraction
  - named entity recognition
  - relation extraction

## Project files

- `code.py` – main script containing the full pipeline
- `paper_embeddings.npy` – saved paper embeddings
- `paper_faiss.index` – saved FAISS index

## Requirements

Install the required Python packages:

```bash
pip install numpy pandas faiss-cpu networkx matplotlib spacy datasets sentence-transformers scikit-learn transformers keybert
```

Download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

## How to run

```bash
python code.py
```

The script will:

1. Download the dataset
2. Generate or load embeddings
3. Create or load the FAISS index
4. Run a sample semantic search
5. Analyze a selected paper with summarization and NLP extraction

## Notes

- The first run may take some time because the script downloads datasets and pretrained models.
- A GPU is not required, but it can significantly improve performance.
- The script may use a large amount of disk space because of model downloads.

## Example output

The script prints:

- dataset size
- embedding shape
- FAISS index size
- top matching papers for a sample query
- generated summaries and extracted keywords/entities/relations


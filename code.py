
import os
import numpy as np
import pandas as pd
import faiss
import networkx as nx
import matplotlib.pyplot as plt
import spacy

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



print("="*70)
print("AI RESEARCH PAPER INTELLIGENT SYSTEM")
print("="*70)


print("\nLoading Dataset...")

dataset = load_dataset(
    "CShorten/ML-ArXiv-Papers",
    split="train"
)

df = pd.DataFrame(dataset)

print("Original Dataset :", len(df))

df = df.head(15000)

print("Using Papers :", len(df))


df["Paper_Text"] = (
    df["title"] +
    " " +
    df["abstract"]
)

df["Paper_Text"] = (
    df["Paper_Text"]
    .str.replace("\n", " ", regex=False)
    .str.strip()
)


print("\nLoading Sentence Transformer...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


sample = df["Paper_Text"].head(5).tolist()

sample_embeddings = model.encode(sample)

print("\nSample Similarities")

for i in range(5):

    score = cosine_similarity(
        sample_embeddings[0].reshape(1,-1),
        sample_embeddings[i].reshape(1,-1)
    )

    print(f"Paper 0 vs Paper {i} :", score[0][0])


if os.path.exists("paper_embeddings.npy"):

    print("\nLoading Saved Embeddings...")

    embeddings = np.load(
        "paper_embeddings.npy"
    )

else:

    print("\nGenerating Embeddings...")

    embeddings = model.encode(

        df["Paper_Text"].tolist(),

        batch_size=32,

        show_progress_bar=True

    )

    np.save(

        "paper_embeddings.npy",

        embeddings

    )

print("Embedding Shape :", embeddings.shape)



faiss.normalize_L2(embeddings)


if os.path.exists("paper_faiss.index"):

    print("\nLoading Existing FAISS Index")

    index = faiss.read_index(
        "paper_faiss.index"
    )

else:

    print("\nCreating FAISS Index")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    faiss.write_index(
        index,
        "paper_faiss.index"
    )

print("Indexed Papers :", index.ntotal)


def search_papers(query, k=5):

    query_embedding = model.encode([query])

    faiss.normalize_L2(query_embedding)

    D, I = index.search(query_embedding, k)

    results = []

    for score, idx in zip(D[0], I[0]):

        results.append({

            "score": float(score),

            "title": df.iloc[idx]["title"],

            "abstract": df.iloc[idx]["abstract"],

            "index": idx

        })

    return results



results = search_papers(

    "Deep Learning in Medical Imaging",

    5

)

print("\nTop Papers\n")

for i, paper in enumerate(results,1):

    print("="*60)

    print("Rank :", i)

    print("Similarity :", round(paper["score"],4))

    print("Title :", paper["title"])

    print("Abstract :")

    print(paper["abstract"][:400])

    print()



from transformers import pipeline
from keybert import KeyBERT


print("\nLoading Summarizer...")

from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)


print("Loading KeyBERT...")

kw_model = KeyBERT(model)


print("Loading spaCy Model...")

nlp = spacy.load("en_core_web_sm")

from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def summarize_paper(text):

    if len(text) < 150:
        return text

    summary = summarizer(
        text,
        max_length=120,
        min_length=40,
        do_sample=False
    )

    return summary[0]["summary_text"]




def extract_keywords(text):

    keywords = kw_model.extract_keywords(

        text,

        keyphrase_ngram_range=(1,2),

        stop_words="english",

        top_n=10

    )

    return keywords


def extract_entities(text):

    doc = nlp(text)

    entities = []

    for ent in doc.ents:

        entities.append({

            "Entity": ent.text,

            "Type": ent.label_

        })

    return entities



def extract_relations(text):

    doc = nlp(text)

    relations = []

    for sent in doc.sents:

        subject = None
        relation = None
        obj = None

        for token in sent:

            if "subj" in token.dep_:
                subject = token.text

            elif token.dep_ == "ROOT":
                relation = token.lemma_

            elif "obj" in token.dep_:
                obj = token.text

        if subject and relation and obj:

            relations.append({

                "Subject": subject,

                "Relation": relation,

                "Object": obj

            })

    return relations


def display_paper_analysis(result):

    print("\n")
    print("="*80)

    print("TITLE\n")
    print(result["title"])

    print("\nSimilarity Score :", round(result["score"],4))

    print("\nABSTRACT\n")
    print(result["abstract"][:800])

    print("\nGenerating Summary...\n")

    summary = summarize_paper(result["abstract"])

    print(summary)

    print("\n")

    print("="*80)

    print("KEYWORDS")

    print("="*80)

    keywords = extract_keywords(result["abstract"])

    for k, score in keywords:

        print(f"{k}   ({round(score,3)})")

    print("\n")

    print("="*80)

    print("NAMED ENTITIES")

    print("="*80)

    entities = extract_entities(result["abstract"])

    if len(entities)==0:

        print("No entities found.")

    else:

        for entity in entities:

            print(entity["Entity"], " --> ", entity["Type"])

    print("\n")

    print("="*80)

    print("RELATIONS")

    print("="*80)

    relations = extract_relations(result["abstract"])

    if len(relations)==0:

        print("No relations extracted.")

    else:

        for r in relations:

            print(

                r["Subject"],

                "--",

                r["Relation"],

                "-->",

                r["Object"]

            )

    return relations




def build_knowledge_graph(relations):

    if len(relations) == 0:
        print("\nNo relations available to create graph.")
        return

    G = nx.DiGraph()

    for r in relations:

        subject = r["Subject"]
        relation = r["Relation"]
        obj = r["Object"]

        G.add_node(subject)
        G.add_node(obj)

        G.add_edge(subject, obj, label=relation)

    plt.figure(figsize=(12,8))

    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=3000,
        font_size=10,
        arrows=True
    )

    edge_labels = nx.get_edge_attributes(G, "label")

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=9
    )

    plt.title("Knowledge Graph")

    plt.show()




def detect_research_gap(text):

    gap_keywords = [

        "future work",
        "limitation",
        "limitations",
        "challenge",
        "challenges",
        "however",
        "although",
        "problem remains",
        "further research",
        "open problem",
        "still difficult",
        "lack",
        "insufficient"

    ]

    found = []

    text = text.lower()

    for word in gap_keywords:

        if word in text:
            found.append(word)

    return found



def future_work_generator(text):

    suggestions = []

    lower = text.lower()

    if "accuracy" in lower:
        suggestions.append(
            "Improve prediction accuracy using ensemble or transformer models."
        )

    if "small dataset" in lower:
        suggestions.append(
            "Validate the approach on larger datasets."
        )

    if "deep learning" in lower:
        suggestions.append(
            "Experiment with foundation models and Vision Transformers."
        )

    if "medical" in lower:
        suggestions.append(
            "Evaluate the model on multi-hospital datasets."
        )

    if "real time" in lower:
        suggestions.append(
            "Optimize inference for real-time deployment."
        )

    if len(suggestions)==0:

        suggestions.append(
            "Investigate larger datasets, multimodal learning and transformer architectures."
        )

    return suggestions



def analyze_paper(result):

    relations = display_paper_analysis(result)

    print("\n")
    print("="*80)
    print("RESEARCH GAP")
    print("="*80)

    gaps = detect_research_gap(result["abstract"])

    if len(gaps)==0:

        print("No obvious research gap keywords found.")

    else:

        for g in gaps:
            print("-",g)

    print("\n")

    print("="*80)
    print("FUTURE WORK")
    print("="*80)

    future = future_work_generator(result["abstract"])

    for item in future:
        print("-",item)

    print("\n")

    print("="*80)
    print("KNOWLEDGE GRAPH")
    print("="*80)

    build_knowledge_graph(relations)




from langchain_huggingface import HuggingFacePipeline
from langchain_core.tools import tool



@tool
def search_paper_tool(query: str) -> str:
    """
    Search research papers and return top result.
    """

    results = search_papers(query, 1)

    if len(results) == 0:
        return "No paper found."

    paper = results[0]

    return f"""
Title:
{paper['title']}

Abstract:
{paper['abstract']}
"""





def search_and_analyze(query, k=5):

    print("\nSearching...\n")

    papers = search_papers(query, k)

    if len(papers) == 0:

        print("No paper found.")

        return

    print(f"{len(papers)} papers found.\n")

    for i, paper in enumerate(papers):

        print("="*100)

        print(f"PAPER {i+1}")

        print("="*100)

        analyze_paper(paper)




def ask_llm(text):

    print("\nGenerating Response...\n")

    response = generator(
    text,
    max_new_tokens=200,
    do_sample=False
    )

    print(response[0]["generated_text"])

    print(response)



def main():

    print("\n")
    print("="*70)
    print("AI RESEARCH PAPER INTELLIGENT SYSTEM")
    print("="*70)

    while True:

        print("\n")

        print("1. Search Research Paper")

        print("2. Ask LLM")

        print("3. Exit")

        choice = input("\nEnter Choice : ")

        if choice == "1":

            query = input("\nEnter Research Topic : ")

            k = int(input("Number of Papers : "))

            search_and_analyze(query, k)

        elif choice == "2":

            text = input("\nEnter Text : ")

            ask_llm(text)

        elif choice == "3":

            print("\nThank you!")

            break

        else:

            print("\nInvalid Choice.")


if __name__ == "__main__":

    main()
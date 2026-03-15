import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import BASE_MODEL, MY_MODEL, HF_TOKEN

def processing_data():
    """
    Reads the CSV file and converts each row into a descriptive sentence.
    Each sentence represents one row in the dataset with all columns labeled.
    """
    df = pd.read_csv('data_cleaning/bps_clean.csv')

    # Convert each row into a single sentence like "A: value; B: value; C: value"
    chunks = [
        "; ".join(f"{col}: {row[col]}" for col in df.columns)
        for _, row in df.iterrows()
    ]
    return chunks  # List of sentences, one per row

def embed(chunks):
    """
    Converts a list of sentences into numerical embeddings using a sentence transformer.
    Each embedding captures the semantic meaning of the sentence.
    """
    model = SentenceTransformer(MY_MODEL)
    chunk_embeddings = model.encode(chunks)
    return chunk_embeddings  # NumPy array of embeddings

def build_faiss_index(chunk_embeddings):
    """
    Builds a FAISS index from the embeddings for fast similarity search.
    Uses L2 (Euclidean) distance for nearest neighbor queries.
    """
    d = chunk_embeddings.shape[1]  # Get the embedding dimension
    index = faiss.IndexFlatL2(d)
    index.add(chunk_embeddings)
    return index


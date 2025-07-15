from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def build_faiss_index(text_chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return model, index, embeddings

def search_topk(query, model, index, text_chunks, k=3):
    q_vec = model.encode([query])
    scores, ids = index.search(np.array(q_vec), k)
    return [text_chunks[i] for i in ids[0]]

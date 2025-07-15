import os
from pathlib import Path
from utils.file_ops import extract_pdf_chunks, extract_text_from_pdf
from utils.chunking import simple_chunk_text
from utils.semantic_search import build_faiss_index, search_topk
from utils.gemini_client import client

from dotenv import load_dotenv
load_dotenv()

DOCUMENTS_DIR = "documents/"
CHUNK_DIR = "data/sample_chunks"
PROMPT_PATH = "prompts/decision_prompt.txt"

def read_prompt():
    with open(PROMPT_PATH, "r") as f:
        return f.read()

def parse_query_with_gemini(query):
    prompt = (
        "Extract as JSON:\n"
        "{ age, gender, procedure, city, policy_duration }\n\nQuery:\n"
        f"{query}"
    )
    response = client.generate_content(prompt)
    return response.text

def run_decision_engine(policy_file, user_query, model, index, chunks_text):
    parsed_query = parse_query_with_gemini(user_query)
    top_clauses = search_topk(user_query, model, index, chunks_text)
    clause_context = "\n".join(top_clauses)
    reasoning_prompt = f"""
{read_prompt()}

User Query (structured): {parsed_query}
Relevant Clauses:
{clause_context}
"""
    response = client.generate_content(reasoning_prompt)
    return response.text

def process_all_policies(user_query):
    results = []
    # Loop through each document in the folder
    for pdf_file in Path(DOCUMENTS_DIR).glob("*.pdf"):
        # Per-file chunking
        chunk_dir = os.path.join(CHUNK_DIR, pdf_file.stem)
        pdf_chunks = extract_pdf_chunks(str(pdf_file), chunk_dir)
        # Per-file text extraction and (optional) finer chunking
        all_chunks_text = []
        for chunk in pdf_chunks:
            text = extract_text_from_pdf(chunk)
            all_chunks_text.extend(simple_chunk_text(text))

        # Build (per-file) semantic search index
        model, index, _ = build_faiss_index(all_chunks_text)

        # Run reasoning for this file
        decision = run_decision_engine(str(pdf_file.name), user_query, model, index, all_chunks_text)

        results.append({
            "policy_file": pdf_file.name,
            "decision_output": decision
        })
    return results

def main():
    user_query = input("Enter your query (e.g., '46M, knee surgery, Pune, 3-month policy'): ").strip()
    results = process_all_policies(user_query)
    print("\n=== Results Per Policy File ===\n")
    for res in results:
        print(f"Policy File: {res['policy_file']}")
        print("Result:\n", res["decision_output"])
        print("-" * 40)

if __name__ == "__main__":
    main()

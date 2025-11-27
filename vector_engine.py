from sentence_transformers import SentenceTransformer, util
import os
import re
from pathlib import Path

# Define the dataset folders you want to scan
DATASET_FOLDERS = ["python_dataset", "sql_dataset", "typescript_dataset"]

def get_dataset_root():
    """
    Intelligently finds the root directory containing the datasets.
    Checks current dir, parent dir, and generic project roots.
    """
    # Start from the directory where this script is located
    current_path = Path(__file__).resolve().parent
    
    # Check up to 2 levels up to find the datasets
    for _ in range(3):
        # Check if *all* expected folders exist here
        if all((current_path / folder).exists() for folder in DATASET_FOLDERS):
            return current_path
        current_path = current_path.parent
        
    # Fallback: Just return cwd if we can't find them (user might have partial data)
    return Path.cwd()

def extract_question_from_file(file_path):
    """
    Reads a file and extracts text following 'Question:'.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Regex to find Question line
            match = re.search(r"Question:\s*(.*)", content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

def load_documents():
    """
    Scans all defined dataset folders and returns a list of dictionaries.
    Each dict contains: {'file': filename, 'path': full_path, 'question': question_text}
    """
    root_dir = get_dataset_root()
    documents = []
    
    print(f"Scanning datasets in: {root_dir}")
    
    for folder_name in DATASET_FOLDERS:
        folder_path = root_dir / folder_name
        if not folder_path.exists():
            print(f"⚠️ Warning: Folder not found: {folder_name}")
            continue
            
        # Iterate over .txt files in this folder
        for file_path in folder_path.glob("*.txt"):
            question = extract_question_from_file(file_path)
            if question:
                documents.append({
                    "file": file_path.name,
                    "folder": folder_name,
                    "path": str(file_path),
                    "question": question
                })
    
    print(f"✅ Loaded {len(documents)} documents from {len(DATASET_FOLDERS)} sources.")
    return documents

def search_documents(query, documents, model):
    """
    Performs vector search and returns top N results.
    """
    if not documents:
        return []

    # 1. Encode the user query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # 2. Encode all document questions (This is fast for <1000 docs, but should be cached in prod)
    doc_texts = [doc['question'] for doc in documents]
    doc_embeddings = model.encode(doc_texts, convert_to_tensor=True)

    # 3. Calculate Cosine Similarity
    # resulting scores is a tensor of shape (1, num_docs)
    cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0]

    # 4. Combine results with scores
    results = []
    for i, score in enumerate(cosine_scores):
        doc = documents[i]
        results.append({
            **doc,
            "score": score.item()
        })

    # 5. Sort by score descending
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    return results
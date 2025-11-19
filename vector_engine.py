# from sentence_transformers import SentenceTransformer, util
# import os
# import sys
# from pathlib import Path
# import re # Import regex for pattern matching

# # Set a higher recursion limit for torch operations if needed, though usually unnecessary here
# sys.setrecursionlimit(2000)

# print("Current working directory:", os.getcwd())

# # ---- UTILITY FUNCTION TO EXTRACT KEY DATA ----

# def extract_key_data_from_file(file_path):
#     """
#     Reads a file to extract the Topic, Question, and creates a unique key 
#     (Topic + Question) for duplication checks.
    
#     Returns a tuple (topic_text, question_text, unique_key) or a tuple of None.
#     """
#     topic_text = None
#     question_text = None
#     try:
#         with open(file_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 line = line.strip()
                
#                 # Find Topic
#                 match_topic = re.match(r"Topic:\s*(.*)", line)
#                 if match_topic and topic_text is None:
#                     topic_text = match_topic.group(1).strip()
                    
#                 # Find Question
#                 match_question = re.match(r"Question:\s*(.*)", line)
#                 if match_question and question_text is None:
#                     question_text = match_question.group(1).strip()
                
#                 # Exit early if both found to save I/O
#                 if topic_text and question_text:
#                     unique_key = f"Topic: {topic_text} | Question: {question_text}"
#                     return topic_text, question_text, unique_key

#     except Exception as e:
#         print(f"Error reading file {file_path}: {e}")
        
#     return None, None, None # Fallback if data is missing or error occurred

# # ---- 1️⃣ Define all dataset folder names ----
# # NEW: List of all folders to process
# DATASET_FOLDER_NAMES = ["typescript_dataset", "python_dataset", "sql_dataset"]


# # ---- 2️⃣ Processing function for a single folder ----

# def process_dataset_folder(folder_name, query_embedding, model, unique_questions):
#     """
#     Processes all .txt files in a single dataset folder, calculates similarity 
#     scores against the query, and returns a list of results and a list of duplicate warnings.
#     """
#     print(f"\n--- Processing folder: {folder_name} ---")
    
#     # Construct the folder path
#     folder_path = Path(os.path.join(os.getcwd(), folder_name))

#     if not folder_path.is_dir():
#         print(f"❌ Folder not found at: {folder_path}. Skipping.")
#         return [], [] # Return empty list for similarities and warnings

#     dataset_files = [f.name for f in folder_path.iterdir() if f.name.endswith(".txt")]
#     if not dataset_files:
#         print(f"⚠️ No .txt files found in '{folder_name}'. Skipping.")
#         return [], [] # Return empty list for similarities and warnings
        
#     print(f"✅ Found {len(dataset_files)} files in '{folder_name}'.")

#     similarities = []
#     duplicate_warnings = [] # List to collect warnings instead of printing them immediately
    
#     for file in dataset_files:
#         file_path = folder_path / file
        
#         # NEW: Extract Topic, Question, and Unique Key
#         topic_text, question_text, unique_key = extract_key_data_from_file(file_path)

#         if not (topic_text and question_text):
#             # Skip file if essential data is missing
#             continue

#         # --- REDUNDANCY CHECK (Now based on Topic + Question) ---
#         if unique_key in unique_questions:
#             # Collect the warning message detail, showing Topic for context
#             duplicate_warnings.append(f"File: {file} | Topic: '{topic_text}' | Question: '{question_text[:30]}...'")
#         else:
#             unique_questions.add(unique_key)
#         # ----------------------------
        
#         try:
#             # Encode the extracted question text (Question Text is still the basis for similarity)
#             embedding_text = model.encode(question_text, convert_to_tensor=True)
            
#             # Calculate the cosine similarity score
#             score = util.cos_sim(query_embedding, embedding_text).item()
            
#             # Store the result including the Topic Text for final output clarity
#             similarities.append((question_text, file, score, folder_name, topic_text))
                
#         except Exception as e:
#             print(f"Error processing file {file} in {folder_name}: {e}")
#             continue

#     return similarities, duplicate_warnings # Return both lists


# # ---- 3️⃣ Load the embedding model ----
# print("Loading embedding model...")
# model = SentenceTransformer('all-MiniLM-L6-v2')


# # ---- 4️⃣ Define the query you want to search ----
# # Using a more general query to pull relevant results from all datasets
# query = "How to fix common programming errors?" 

# # ---- 5️⃣ Encode the query ----
# print(f"Encoding search query: '{query}'")
# embedding_query = model.encode(query, convert_to_tensor=True)


# # ---- 6️⃣ Process all datasets and collect results ----
# all_similarities = []
# unique_questions = set() # Set to track unique (Topic + Question) content
# all_duplicate_warnings = {} # Dictionary to store warnings by folder

# for folder_name in DATASET_FOLDER_NAMES:
#     # Capture both the similarities and the warnings
#     results, warnings = process_dataset_folder(folder_name, embedding_query, model, unique_questions) 
    
#     # Add similarities
#     all_similarities.extend(results)
    
#     # Store warnings by folder if any were found
#     if warnings:
#         all_duplicate_warnings[folder_name] = warnings


# # ---- 7️⃣ Sort and show top 10 overall results ----
# top_results = sorted(all_similarities, key=lambda x: x[2], reverse=True)[:10]

# print("\n--- 📊 Top Matching Study Questions Across All Datasets ---")
# print(f"Search Query: {query}")
# print("-" * 70)

# if top_results:
#     # The tuple now contains: (question, file, score, folder_name, topic_text)
#     for question, file, score, folder_name, topic_text in top_results: 
#         # Print the extracted question and its score
#         print(f"  • Dataset: {folder_name:<18} | File: {file:<15} | Similarity: {score:.4f}")
#         print(f"    Topic: {topic_text}")
#         print(f"    Question: {question}")
#         print("-" * 70)
# else:
#     print("No similarity results processed (check file content or folder structure).")


# # --- NEW: Print Warning Summary ---
# if all_duplicate_warnings:
#     print("\n--- ⚠️ Summary of Duplicate Question Content (Unique Topic + Question) ⚠️ ---")
#     for folder, warnings in all_duplicate_warnings.items():
#         print(f"Dataset: {folder:<18} | Total Duplicates Found: {len(warnings)}")
#         # Print only the first 5 examples to keep output clean
#         for i, warning in enumerate(warnings[:5]):
#             # The warning string now contains Topic for better context
#             print(f"  Example {i+1}: {warning}")
#         if len(warnings) > 5:
#             print(f"  ... {len(warnings) - 5} more duplicates omitted for brevity.")
#     print("-" * 70)
# # ------------------------------------


# # ---- 8️⃣ Optional: Show vector snippet ----
# print("\n🧠 Sample Vector Representation (first 100 values of query embedding):")
# print(embedding_query.cpu().numpy()[:100])

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
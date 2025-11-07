from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import os
import re
import sys

sys.setrecursionlimit(2000)

MODEL = SentenceTransformer('all-MiniLM-L6-v2') 
DATASET_FOLDER_NAMES = ["typescript_dataset", "python_dataset", "sql_dataset"]

def extract_key_data_from_file(file_path):
    topic_text = None
    question_text = None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                match_topic = re.match(r"Topic:\s*(.*)", line)
                if match_topic and topic_text is None:
                    topic_text = match_topic.group(1).strip()
                match_question = re.match(r"Question:\s*(.*)", line)
                if match_question and question_text is None:
                    question_text = match_question.group(1).strip()
                if topic_text and question_text:
                    unique_key = f"Topic: {topic_text} | Question: {question_text}"
                    return topic_text, question_text, unique_key
    except Exception:
        pass
    return None, None, None

def process_dataset_folder(folder_name, query_embedding, model, unique_questions):
    folder_path = Path(os.path.join(os.getcwd(), folder_name))
    if not folder_path.is_dir():
        return [], []
    dataset_files = [f.name for f in folder_path.iterdir() if f.name.endswith(".txt")]
    similarities = []
    for file in dataset_files:
        file_path = folder_path / file
        topic_text, question_text, unique_key = extract_key_data_from_file(file_path)
        if not (topic_text and question_text):
            continue
        if unique_key not in unique_questions:
            unique_questions.add(unique_key)
        try:
            embedding_text = model.encode(question_text, convert_to_tensor=True)
            score = util.cos_sim(query_embedding, embedding_text).item()
            similarities.append((question_text, file, score, folder_name, topic_text))
        except Exception:
            continue
    return similarities, []

def get_top_k_documents(query, k=5):
    query_embedding = MODEL.encode(query, convert_to_tensor=True)
    all_similarities = []
    unique_questions = set()
    for folder_name in DATASET_FOLDER_NAMES:
        results, _ = process_dataset_folder(folder_name, query_embedding, MODEL, unique_questions)
        all_similarities.extend(results)
    top_results = sorted(all_similarities, key=lambda x: x[2], reverse=True)[:k]
    if not top_results:
        return "No relevant documents were found in the knowledge base."
    context_list = []
    for question, _, score, folder_name, topic_text in top_results:
        context_list.append(
            f"Source: {folder_name} (Topic: {topic_text}, Score: {score:.3f})\n"
            f"Question/Fact: {question}"
        )
    return "\n---\n".join(context_list)

import streamlit as st
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import vector_engine  # retrieval logic
import functions      # generation logic

# --- Configuration ---
st.set_page_config(page_title="Smart Study Assistant", layout="wide")

# Load API Key from secrets or environment
# ideally use st.secrets["GOOGLE_API_KEY"] for deployment
API_KEY = os.getenv("GOOGLE_API_KEY") 
if not API_KEY:
    st.error("Missing Google API Key. Please set GOOGLE_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=API_KEY)

# --- 1. Initialize Resources ---

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_gemini_model():
    # Configure the Gemini model once
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
    }
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

@st.cache_data
def load_knowledge_base():
    return vector_engine.load_documents()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the chat session object required by functions.py
if "chat_session" not in st.session_state:
    model = get_gemini_model()
    st.session_state.chat_session = model.start_chat(history=[])

# --- 2. Load Data & Models ---
try:
    embed_model = load_embedding_model()
    documents = load_knowledge_base()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

# --- 3. Sidebar ---
with st.sidebar:
    st.header("Study Settings")
    st.write(f"📚 **Loaded Documents:** {len(documents)}")
    
    # Technology Filter
    selected_tech = st.multiselect(
        "Focus Areas",
        options=["python_dataset", "sql_dataset", "typescript_dataset"],
        default=["python_dataset", "sql_dataset", "typescript_dataset"]
    )
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_session = get_gemini_model().start_chat(history=[])
        st.rerun()

# --- 4. Main Chat Interface ---
st.title("🎓 Smart Study Engine")
st.caption("Ask questions about Python, SQL, or TypeScript. I'll answer using your study notes.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
user_query = st.chat_input("Ask a study question...")

if user_query:
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. Filter Knowledge Base
    filtered_docs = [d for d in documents if d['folder'] in selected_tech]

    # 3. RETRIEVAL: Find relevant context
    with st.status("Consulting knowledge base...", expanded=False) as status:
        st.write("Searching vector space...")
        relevant_docs = vector_engine.search_documents(user_query, filtered_docs, embed_model)
        
        # Take top 3 most relevant documents
        top_docs = relevant_docs[:3]
        
        # Construct Context String
        context_text = ""
        for doc in top_docs:
            # Load the actual content of the file to pass to Gemini
            try:
                with open(doc['path'], "r", encoding="utf-8") as f:
                    file_content = f.read()
                    context_text += f"\n\n--- Document: {doc['file']} ({doc['folder']}) ---\n{file_content}"
            except Exception as e:
                st.warning(f"Could not read context file {doc['file']}")

        status.update(label="Knowledge retrieved!", state="complete", expanded=False)

    # 4. DISPLAY SOURCES ONLY (No Gemini Generation)
    with st.chat_message("assistant"):
        st.info("Here are the most relevant study materials found:")
        
        for doc in top_docs:
            score = doc['score']
            color = "green" if score > 0.5 else "orange"
            
            with st.expander(f"{doc['file']} (Score: :{color}[{score:.4f}])", expanded=True):
                st.markdown(f"**Question:** {doc['question']}")
                try:
                    with open(doc['path'], "r", encoding="utf-8") as f:
                        st.code(f.read(), language=doc['folder'].replace("_dataset", ""))
                except Exception as read_err:
                    st.error(f"Could not read file content: {read_err}")

        # --- GEMINI GENERATION SECTION (DISABLED) ---
        # response_text = functions.fetch_gemini_response(user_query, context=context_text)
        # st.markdown(response_text)
        # st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Since we aren't generating a new response to save to history, we might want to 
    # save a placeholder or simply not append to history if strictly viewing docs.
    # For now, we won't append an 'assistant' message to history to avoid clutter 
    # or we can append a summary of what was found.
    st.session_state.messages.append({"role": "assistant", "content": "I've displayed the relevant study documents above."})
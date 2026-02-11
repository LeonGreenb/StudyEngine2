import google.generativeai as genai

def map_role(role):
    if role == "model":
        return "assistant"
    else:
        return role

def fetch_gemini_response(chat_session, user_query, context=""):
    """
    Generates a response from Gemini using the provided chat session.
    
    Args:
        chat_session: The Gemini ChatSession object (e.g., st.session_state.chat_session).
        user_query (str): The user's question.
        context (str): The retrieved knowledge base content.
    """
    system_instruction = (
        "You are the **Study Engine**, an expert tutor. Use the provided context documents "
        "to answer the user's question accurately and thoroughly. "
        "If the context does not contain the answer, state clearly that you could not "
        "find the specific information in the provided knowledge base. Do not mention the context score in your final answer."
    )
    
    full_prompt = (
        f"{system_instruction}"
        "\n\n--- KNOWLEDGE BASE CONTEXT ---\n"
        f"{context}"
        "\n\n--- USER QUESTION ---\n"
        f"{user_query}"
    )
    
    try:
        response = chat_session.send_message(full_prompt)
        # print(f"Gemini's Response: {response.text}") # Optional logging
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"
import streamlit as st

def map_role(role):
    if role == "model":
        return "assistant"
    else:
        return role

def fetch_gemini_response(user_query, context=""):
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
    response = st.session_state.chat_session.model.generate_content(full_prompt)
    print(f"Gemini's Response: {response}")
    return response.parts[0].text
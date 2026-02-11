# 💬 Chatbot template

A simple Streamlit app that shows how to build a chatbot using OpenAI's GPT-3.5.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run app.py
   ```

# Other option 
1. Create the virtual environment
   `python3 -m venv .venv`

2. Activate the environment:
   `source .venv/bin/activate`

3. Install your packages:
   `pip install streamlit sentence-transformers google-generativeai torch`

4. Set your Google API Key
   `export GOOGLE_API_KEY="AIzaSy..."`

5. Run the Application
   `streamlit run app.py`

This should open a new tab in the default web browser (usually at http://localhost:8501) displaying the Smart Study Assistant interface. You can now test the search functionality by typing a query like "SQL joins" or "Python lists" into the chat box.
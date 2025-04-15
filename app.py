import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from googletrans import LANGCODES
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up language codes
languages = {v: k for k, v in LANGCODES.items()}
language_list = list(languages.keys())

def translate_text(text, source_lang, target_lang, model_type):
    # Create appropriate prompt
    prompt = ChatPromptTemplate.from_template(
        """You are a professional translator. Translate the following text from {source_lang} to {target_lang}.
        Maintain the meaning, tone, and context exactly. Only respond with the translation.
        
        Text: {text}"""
    )
    
    # Initialize the selected model
    if model_type == "Groq":
        llm = ChatGroq(
            model_name="mixtral-8x7b-32768",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1
        )
    else:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        )
    
    # Create chain
    chain = prompt | llm
    
    # Handle auto-detect
    if source_lang == "Auto-Detect":
        source_lang = "the detected source language"
    
    # Invoke chain
    response = chain.invoke({
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang
    })
    
    return response.content

# Streamlit UI
st.title("Multi-Language Translator 🌍")

# Create columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    input_text = st.text_area("Enter text to translate:", height=200)
    source_lang = st.selectbox(
        "Source Language (Optional):",
        ["Auto-Detect"] + language_list,
        index=0
    )

with col2:
    st.subheader("Output")
    target_lang = st.selectbox(
        "Target Language:",
        language_list,
        index=language_list.index("English") if "English" in language_list else 0
    )
    
    # Model selection
    model_type = st.radio(
        "Translation Model:",
        ["Groq (Mixtral)", "Google Gemini"],
        horizontal=True
    )
    
    if st.button("Translate ✨"):
        if input_text:
            with st.spinner("Translating..."):
                try:
                    translation = translate_text(
                        input_text,
                        source_lang,
                        target_lang,
                        "Groq" if model_type == "Groq (Mixtral)" else "Google"
                    )
                    st.text_area("Translation:", value=translation, height=200)
                except Exception as e:
                    st.error(f"Error in translation: {str(e)}")
        else:
            st.warning("Please enter some text to translate")
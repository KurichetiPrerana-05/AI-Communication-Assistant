import streamlit as st
import pandas as pd
from transformers import pipeline
import spacy
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import plotly.express as px

st.set_page_config(page_title="AI Communication Assistant", layout="wide")

@st.cache_resource
def load_models():
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    nlp = spacy.load("en_core_web_sm")
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
    generator_pipeline = pipeline('text-generation', model='gpt2', max_new_tokens=120)
    return sentiment_pipeline, nlp, retriever_model, generator_pipeline

@st.cache_resource
def setup_rag():
    with open("knowledge_base.txt", "r") as f:
        knowledge_base = [line.strip() for line in f.readlines()]
    kb_embeddings = retriever_model.encode(knowledge_base, convert_to_tensor=True)
    index = faiss.IndexFlatL2(kb_embeddings.shape[1])
    index.add(kb_embeddings.cpu().numpy())
    return knowledge_base, index

st.title("📧 AI-Powered Communication Assistant")

# --- Load Models and RAG ---
with st.spinner("Loading AI models... This may take a moment."):
    sentiment_pipeline, nlp, retriever_model, generator_pipeline = load_models()
    knowledge_base, faiss_index = setup_rag()

# --- Load Data ---
try:
    df = pd.read_csv("Sample_Support_Emails_Dataset.csv")
except FileNotFoundError:
    st.error("`Sample_Support_Emails_Dataset.csv` not found. Please place it in the project folder.")
    st.stop()

# --- Your AI Functions (Copied from Colab) ---
def process_emails(df):
    filter_keywords = ['support', 'query', 'request', 'help', 'urgent', 'critical']
    df['is_support_ticket'] = df['subject'].str.contains('|'.join(filter_keywords), case=False)
    support_df = df[df['is_support_ticket']].copy()
    if support_df.empty: return pd.DataFrame()
    support_df['sentiment'] = support_df['body'].apply(lambda x: sentiment_pipeline(x[:512])[0]['label'])
    urgent_keywords = ['urgent', 'critical', 'immediately', 'cannot access', 'down', 'blocked']
    support_df['priority'] = support_df.apply(
        lambda row: 'Urgent' if any(kw in (row['subject'] + ' ' + row['body']).lower() for kw in urgent_keywords) else 'Not Urgent',
        axis=1
    )
    return support_df.sort_values(by='priority', ascending=False).reset_index(drop=True)

def generate_response_rag(email_body, email_subject):
    query_embedding = retriever_model.encode(email_body, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
    _, indices = faiss_index.search(query_embedding, 1)
    retrieved_context = knowledge_base[indices[0][0]]
    prompt = f"Context: \"{retrieved_context}\"\n\nCustomer Email: \"{email_body}\"\n\nDraft a helpful and professional response:"
    generated_output = generator_pipeline(prompt)[0]['generated_text']
    return generated_output.split("Draft a helpful and professional response:")[-1].strip()

# --- Build the Dashboard ---
processed_df = process_emails(df)

st.sidebar.header("Analytics")
st.sidebar.metric("Total Filtered Emails", len(processed_df))
st.sidebar.metric("🔥 Urgent Tickets", (processed_df['priority'] == 'Urgent').sum())

st.header("Support Email Queue")
for index, row in processed_df.iterrows():
    with st.expander(f"{'🔥' if row['priority'] == 'Urgent' else '✉️'} **{row['subject']}** - Priority: {row['priority']}"):
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**From:** {row['sender']}\n\n**Sentiment:** {row['sentiment']}")
            st.write(row['body'])
        with col2:
            if st.button("Generate AI Response", key=f"btn_{index}"):
                response = generate_response_rag(row['body'], row['subject'])
                st.text_area("AI-Generated Response:", value=response, height=250, key=f"txt_{index}")
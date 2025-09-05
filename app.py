import streamlit as st
import pandas as pd
from transformers import pipeline
import spacy
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import plotly.express as px
import sqlite3
import re
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Communication Assistant",
    page_icon="🤖",
    layout="wide",
)

# --- Database Setup ---
@st.cache_resource
def init_db():
    """Initialize the SQLite database and create tables."""
    conn = sqlite3.connect("support_emails.db", check_same_thread=False)
    c = conn.cursor()
    # Create table for processed emails
    c.execute('''
        CREATE TABLE IF NOT EXISTS emails (
            id INTEGER PRIMARY KEY,
            sender TEXT,
            subject TEXT,
            body TEXT,
            priority TEXT,
            sentiment TEXT,
            intent TEXT,
            received_date TIMESTAMP
        )
    ''')
    # Create table for response templates
    c.execute('''
        CREATE TABLE IF NOT EXISTS templates (
            id INTEGER PRIMARY KEY,
            title TEXT UNIQUE,
            body TEXT
        )
    ''')
    conn.commit()
    return conn

db_conn = init_db()

# --- AI Model Loading ---
@st.cache_resource
def load_models():
    """Load all AI models once."""
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    # Zero-shot for intent classification
    intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    nlp = spacy.load("en_core_web_sm")
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
    generator_pipeline = pipeline('text-generation', model='gpt2', max_new_tokens=150)
    return sentiment_pipeline, intent_classifier, nlp, retriever_model, generator_pipeline

# --- RAG Setup ---
@st.cache_resource
def setup_rag():
    """Load KB, create embeddings, and build FAISS index."""
    with open("knowledge_base.txt", "r") as f:
        knowledge_base = [line.strip() for line in f.readlines()]
    kb_embeddings = retriever_model.encode(knowledge_base, convert_to_tensor=True)
    index = faiss.IndexFlatL2(kb_embeddings.shape[1])
    index.add(kb_embeddings.cpu().numpy())
    return knowledge_base, index

# --- Core AI & DB Functions ---
def get_intent(text):
    """Classify the intent of the email."""
    candidate_labels = ["Billing Issue", "Login Problem", "Technical Bug", "Feature Request", "General Inquiry", "Account Verification"]
    result = intent_classifier(text[:512], candidate_labels)
    return result['labels'][0]

def extract_entities(text):
    """Extract actionable entities like invoice numbers."""
    # Simple regex for invoice numbers (e.g., INV-12345, Order# 98765)
    invoice_pattern = r'(invoice|order)\s*#?\s*([A-Z0-9-]+)'
    match = re.search(invoice_pattern, text, re.IGNORECASE)
    if match:
        return {"Invoice/Order #": match.group(2)}
    return "No specific entities found"

def save_template(title, body):
    """Save a new response template to the database."""
    try:
        c = db_conn.cursor()
        c.execute("INSERT INTO templates (title, body) VALUES (?, ?)", (title, body))
        db_conn.commit()
        st.sidebar.success(f"Template '{title}' saved!")
    except sqlite3.IntegrityError:
        st.sidebar.error(f"Template title '{title}' already exists.")

def load_templates():
    """Load all templates from the database."""
    c = db_conn.cursor()
    c.execute("SELECT title, body FROM templates")
    return {title: body for title, body in c.fetchall()}

def log_email_to_db(row):
    """Log a processed email to the database if not already present."""
    c = db_conn.cursor()
    # Check if this exact email body from the sender is already logged
    c.execute("SELECT id FROM emails WHERE sender = ? AND body = ?", (row['sender'], row['body']))
    if c.fetchone() is None:
        c.execute(
            "INSERT INTO emails (sender, subject, body, priority, sentiment, intent, received_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (row['sender'], row['subject'], row['body'], row['priority'], row['sentiment'], row['intent'], row['sent_date'])
        )
        db_conn.commit()

# --- Main Application UI ---
st.title(" Advanced AI Communication Assistant")

with st.spinner("Loading AI models... This may take a moment."):
    sentiment_pipeline, intent_classifier, nlp, retriever_model, generator_pipeline = load_models()
    knowledge_base, faiss_index = setup_rag()

# --- Process and Display Emails ---
df = pd.read_csv("Sample_Support_Emails_Dataset.csv")
df['sent_date'] = pd.to_datetime(df['sent_date'])

# Apply basic filtering first
filter_keywords = ['support', 'query', 'request', 'help', 'urgent', 'critical']
df['is_support_ticket'] = df['subject'].str.contains('|'.join(filter_keywords), case=False)
support_df = df[df['is_support_ticket']].copy()

if not support_df.empty:
    support_df['sentiment'] = support_df['body'].apply(lambda x: sentiment_pipeline(x[:512])[0]['label'])
    urgent_keywords = ['urgent', 'critical', 'immediately', 'cannot access', 'down', 'blocked']
    support_df['priority'] = support_df.apply(
        lambda row: 'Urgent' if any(kw in (row['subject'] + ' ' + row['body']).lower() for kw in urgent_keywords) else 'Not Urgent',
        axis=1
    )
    # --- New Feature: Intent & Entity Recognition ---
    support_df['intent'] = support_df['body'].apply(get_intent)
    support_df['entities'] = support_df['body'].apply(extract_entities)

    # Log new emails to DB
    for _, row in support_df.iterrows():
        log_email_to_db(row)

    support_df = support_df.sort_values(by='priority', ascending=False).reset_index(drop=True)

    # --- Sidebar ---
    st.sidebar.header("Analytics")
    st.sidebar.metric("Total Filtered Emails", len(support_df))
    st.sidebar.metric("🔥 Urgent Tickets", (support_df['priority'] == 'Urgent').sum())

    # --- Main Dashboard ---
    st.header("Prioritized Email Queue")

    for index, row in support_df.iterrows():
        priority_emoji = "🔥" if row['priority'] == 'Urgent' else "✉️"
        with st.expander(f"{priority_emoji} **{row['subject']}** from `{row['sender']}` | Intent: **{row['intent']}**"):
            col1, col2 = st.columns([2, 3])
            with col1:
                st.subheader("Email Details")
                st.markdown(f"**From:** {row['sender']}")
                st.markdown(f"**Priority:** {row['priority']} | **Sentiment:** {row['sentiment']}")
                st.markdown(f"**Extracted Data:** `{row['entities']}`")
                st.info(row['body'])

            with col2:
                st.subheader("AI-Powered Response")
                templates = load_templates()
                if templates:
                    template_choice = st.selectbox("Use a template", ["None"] + list(templates.keys()), key=f"tpl_{index}")
                    if template_choice != "None":
                        st.session_state[f'response_{index}'] = templates[template_choice]

                if st.button("Generate AI Response", key=f"btn_{index}"):
                    with st.spinner(" Generating context-aware response..."):
                        # RAG Implementation
                        query_embedding = retriever_model.encode(row['body'], convert_to_tensor=True).cpu().numpy().reshape(1, -1)
                        _, indices = faiss_index.search(query_embedding, 1)
                        retrieved_context = knowledge_base[indices[0][0]]
                        prompt = f"Context: \"{retrieved_context}\"\n\nCustomer Email: \"{row['body']}\"\n\nDraft a helpful and professional response:"
                        generated_output = generator_pipeline(prompt)[0]['generated_text']
                        response = generated_output.split("Draft a helpful and professional response:")[-1].strip()
                        st.session_state[f'response_{index}'] = response

                response_text = st.text_area("Review and edit response:", value=st.session_state.get(f'response_{index}', ""), height=200, key=f'txt_{index}')

                if response_text:
                    if st.button("Save as Template", key=f"save_{index}"):
                        template_title = st.text_input("Template Title", key=f"title_{index}")
                        if template_title:
                            save_template(template_title, response_text)

# --- Historical Analytics Section ---
st.header("Historical Analytics")
db_df = pd.read_sql_query("SELECT intent, received_date FROM emails", db_conn)
if not db_df.empty:
    db_df['received_date'] = pd.to_datetime(db_df['received_date'])
    intent_counts = db_df.groupby([db_df['received_date'].dt.date, 'intent']).size().reset_index(name='count')
    fig = px.bar(intent_counts, x='received_date', y='count', color='intent', title='Daily Email Intents Over Time')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No historical data to display yet. Processed emails will appear here.")
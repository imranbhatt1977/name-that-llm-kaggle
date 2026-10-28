import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

# Page Config
st.set_page_config(page_title="Name That LLM - Authorship Attribution", page_icon="🤖", layout="wide")

@st.cache_resource
def load_model_and_tokenizer():
    model_name = "microsoft/deberta-v3-base"
    # Note: In a live app, you would load your fine-tuned weights: 
    # model.load_state_dict(torch.load("deberta_v3_fold_0.pth", map_location="cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=18)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Dummy class labels representing your 18 competition models
LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"
]

st.title("🤖 Name That LLM: Authorship Attribution App")
st.markdown("Paste a text passage below to predict which frontier language model or lab generated it!")

# Text Input Area
user_text = st.text_area("Enter text passage (40 - 250 words):", height=150, placeholder="Type or paste text here...")

if st.button("Predict Author Model", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing stylistic fingerprints and transformer probabilities..."):
            # Tokenize input
            inputs = tokenizer(
                user_text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1).numpy()[0]
                
            # Create a DataFrame for visualization
            results_df = pd.DataFrame({
                'Model': LABELS,
                'Probability': probs
            }).sort_values(by='Probability', ascending=False).reset_index(drop=True)
            
            top_prediction = results_df.iloc[0]
            
            # Display Top Prediction
            st.success(f"🎯 **Predicted Model:** `{top_prediction['Model']}` (Confidence: {top_prediction['Probability']*100:.2f}%)")
            
            # Display Full Distribution Chart
            st.subheader("📊 Full Model Probability Distribution")
            st.bar_chart(results_df.set_index('Model'))

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os

st.set_page_config(page_title="SciBERT Semantic Query Engine", page_icon="🔬", layout="centered")

st.title("🔬 Scientific Inference & Decision Engine")
st.markdown("""
This interface runs a fine-tuned **SciBERT** model trained on scientific assertions to evaluate clinical query outcomes.
""")

MODEL_PATH = "./saved_model" if os.path.exists("./saved_model") else "allenai/scibert_scivocab_uncased"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # If using base model on cloud, map to 3 classes
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, 
        num_labels=3,
        id2label={0: "no", 1: "yes", 2: "maybe"},
        label2id={"no": 0, "yes": 1, "maybe": 2}
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

query = st.text_input(
    "Research Question:", 
    value="Do statins reduce cardiovascular events in diabetic patients?"
)
context = st.text_area(
    "Clinical Context / Abstract:", 
    value="Multiple randomized controlled trials show significant reduction in major adverse cardiac events among diabetic cohorts treated with atorvastatin compared to placebo...",
    height=140
)

if st.button("Evaluate Hypothesis", type="primary"):
    input_text = f"Question: {query} Context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = F.softmax(logits, dim=-1).squeeze().tolist()
    
    labels = [model.config.id2label[i] for i in range(len(probabilities))]
    results_df = pd.DataFrame({
        "Conclusion": [l.upper() for l in labels],
        "Probability": probabilities
    }).sort_values(by="Probability", ascending=False).reset_index(drop=True)
    
    top_conclusion = results_df.iloc[0]["Conclusion"]
    top_score = results_df.iloc[0]["Probability"] * 100
    
    st.metric(label="Predicted Assertion", value=top_conclusion, delta=f"{top_score:.1f}% Confidence")
    
    st.write("### Decision Probability Distribution")
    st.bar_chart(results_df.set_index("Conclusion"))

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
import os

model_path = os.path.abspath('D:\Ironhack\week6\Project_NLP_Business_Case\checkpoint750')
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained(model_path , local_files_only=True)  
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
    return model, tokenizer

model, tokenizer = load_model()

# Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    label = torch.argmax(probs, dim=1).item()
    confidence = probs[0][label].item()
    return label, confidence

# Streamlit UI
st.title("BERT Sentiment Classifier")
user_input = st.text_area("Enter a review or comment:")
if st.button("Analyze"):
    if user_input.strip() != "":
        label, confidence = predict_sentiment(user_input)
        sentiment = {0: "Negative", 1: "Neutral", 2: "Positive"}.get(label, "Unknown")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.warning("Please enter some text.")

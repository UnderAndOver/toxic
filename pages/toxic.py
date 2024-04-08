import streamlit as st
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Setup (Loading model from your output directory)
model_name = "output_dir/models"


@st.cache_resource  # Caching to avoid reloading on every interaction
def load_model():
    Bert_Tokenizer = BertTokenizer.from_pretrained(model_name)
    Bert_Model = BertForSequenceClassification.from_pretrained(model_name).to(device)
    return Bert_Model, Bert_Tokenizer


Bert_Model, Bert_Tokenizer = load_model()


def predict_user_input(
    input_text, model=Bert_Model, tokenizer=Bert_Tokenizer, device=device
):
    # covert the colab code to streamlit code
    user_input = [input_text]
    user_encodings = tokenizer(
        user_input, padding=True, truncation=True, return_tensors="pt"
    )
    user_dataset = TensorDataset(
        user_encodings["input_ids"], user_encodings["attention_mask"]
    )
    user_loader = DataLoader(user_dataset, batch_size=1, shuffle=False)

    ban_words = ["X"]
    model.eval()
    with torch.no_grad():
        for batch in user_loader:
            input_ids, attention_mask = [t.to(device) for t in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.sigmoid(logits)

    predicted_labels = (predictions.cpu().numpy() > 0.5).astype(int)
    labels_list = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    result = dict(zip(labels_list, predicted_labels[0]))
    return result


# Streamlit App Interface
st.title("Toxicity Detection App")
user_text = st.text_area("Enter your text:", height=200)

if st.button("Analyze"):
    if user_text:
        with st.spinner("Analyzing..."):
            results = predict_user_input(user_text)
            st.markdown("**Results:**")
            for label, score in results.items():
                st.write(f"{label.capitalize()}: {score}")
    else:
        st.warning("Please enter some text to analyze.")

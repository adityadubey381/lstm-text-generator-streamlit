# LSTM Text Generator (Streamlit)

A simple text generation web app built using PyTorch and Streamlit.  
The model is trained from scratch using an LSTM to predict the next word in a sequence.

## Features
- Custom tokenizer and vocabulary
- LSTM-based language model
- Next-word prediction
- Generates short text sequences
- Simple Streamlit UI

## How It Works
1. User enters a sentence  
2. Text is tokenized and converted to numerical form  
3. Model predicts the next word  
4. Process repeats to generate a sequence  

## Tech Stack
- Python  
- PyTorch  
- Streamlit  

##  Run Locally
```bash
git clone <your-repo-link>
cd lstm-text-generator-streamlit
pip install -r requirements.txt
streamlit run app.py

import torch
import numpy as np
import streamlit as st

from tokenizer import word_tokenization, text_to_indices
from train import padding, LSTMModel

# load vocab
vocab = torch.load("vocab.pth")
idx_to_word = {v: k for k, v in vocab.items()}

# load model
model = LSTMModel(len(vocab))
checkpoint = torch.load("model.pth")
model.load_state_dict(checkpoint["model_state"])
model.eval()

st.title("Text Generator Based on attention all you need paper data")

user_input = st.text_input("Enter sentence")


st.markdown("""
    <style>
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

if st.button("Generate"):

    if user_input.strip() == "":
        st.write("Enter something first")
        st.stop()

    sentences = []

    for _ in range(10):   # 10 sentences

        tokens = word_tokenization(user_input.lower())
        generated = tokens.copy()

        for _ in range(20):   # max words per sentence

            numerical = text_to_indices(generated, vocab)
            padded_seq = padding([numerical])
            input_tensor = torch.tensor(padded_seq).long()

            with torch.no_grad():
                output = model(input_tensor)

            probs = torch.softmax(output, dim=1).squeeze().numpy()

            # 🔥 sampling instead of argmax
            next_idx = np.random.choice(len(probs), p=probs)

            next_word = idx_to_word.get(next_idx, "<UNK>")

            # skip unknown words
            if next_word == "<UNK>":
                continue

            generated.append(next_word)

            # stop condition (optional)
            if next_word == ".":
                break

        sentences.append(" ".join(generated))

    # print all sentences
    for i, s in enumerate(sentences):
        st.write(f"{i+1}. {s}")

# Sidebar or main button
else :
   if st.sidebar.button("About"):

      st.markdown("""
      ## About This Project

      This project is a **Text Generation Web Application** built using **Streamlit** and a custom deep learning model implemented in **PyTorch**.

      It demonstrates how a language model can be trained from scratch to generate text based on user input by predicting the next word in a sequence.

      ---

      ## Model Architecture

      The core model is based on a **Recurrent Neural Network (LSTM)**, which learns sequential dependencies in text data. It processes input text through:

      - Tokenization and vocabulary mapping  
      - Sequence padding  
      - Embedding layer  
      - LSTM layer  
      - Fully connected output layer  

      The model predicts the next word iteratively to generate text.

      ---

      ## Trained On  "Attention Is All You Need"

      This project is conceptually inspired by the paper:
      **"Attention Is All You Need" (Vaswani et al., 2017)**

      While this implementation uses LSTM, the paper introduces the **Transformer architecture**, which replaces recurrence with attention mechanisms and forms the foundation of modern AI systems like GPT.

      This project serves as a stepping stone toward understanding and implementing more advanced architectures like Transformers.

      ---

      ## Features

      - Custom tokenizer and vocabulary  
      - Sequence-based text generation  
      - Iterative next-word prediction  
      - Interactive UI using Streamlit  

      ---

      ## Limitations

      - Limited dataset → lower coherence  
      - LSTM-based → struggles with long-range dependencies  
      - Not comparable to modern LLMs  

      """)
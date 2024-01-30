# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
#importing data
data = pd.read_csv(r"E:\sr.ds\DataNeuron_DataScience_Task\DataNeuron_Text_Similarity.csv")

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to encode a text into BERT embeddings
def encode_text(text):
    # Tokenize input text
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    # Forward pass to get BERT embeddings
    with torch.no_grad():
        outputs = model(**tokens)
    
    # Extract embeddings from the last layer
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return embeddings

# Function to calculate similarity between two sentences using cosine similarity
def calculate_similarity(sentence1, sentence2):
    # Encode both sentences
    embeddings1 = encode_text(sentence1)
    embeddings2 = encode_text(sentence2)

    # Calculate cosine similarity
    similarity_score = 1 - cosine(embeddings1, embeddings2)

    return similarity_score

# Take input from the user
text1 = data['text1'][2304]
text2 = data['text2'][2304]

# Calculate and print similarity score
similarity_score = calculate_similarity(text1, text2)
print(f"Similarity Score: {similarity_score:.4f}")



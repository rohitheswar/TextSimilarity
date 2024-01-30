# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 03:49:46 2024

@author: RohithThokala
"""

from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine

app = Flask(__name__)

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

@app.route('/calculate_similarity', methods=['POST'])
def get_similarity():
    data = request.get_json()
    text1 = data.get('text1')
    text2 = data.get('text2')

    # Calculate similarity
    similarity_score = calculate_similarity(text1, text2)

    # Return result as JSON
    return jsonify({'similarity_score': similarity_score})

if __name__ == '__main__':
    app.run(debug=True)

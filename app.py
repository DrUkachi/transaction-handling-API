import os
from typing import List, Dict, Union, Tuple
from fastapi import FastAPI, HTTPException, Depends
import pandas as pd
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import torch
from fuzzywuzzy import fuzz

# Initialize FastAPI app
app = FastAPI()

# Global variables for data storage
trans_data = None
users_data = None

# Load DistilBERT model and tokenizer for NER
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModel.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to load data components
def load_data_components():
    try:
        cwd = os.getcwd()
        transaction_filename = "transactions.csv"
        users_filename = "users.csv"
        
        transaction_filepath = os.path.join(cwd, transaction_filename)
        users_filepath = os.path.join(cwd, users_filename)
        
        transaction_data = pd.read_csv(transaction_filepath)
        users_data = pd.read_csv(users_filepath)
        
        return transaction_data, users_data
    except Exception as e:
        raise RuntimeError(f"Failed to load Data: {e}")

# Dependency function to ensure data is loaded before endpoint functions
def get_loaded_data():
    global trans_data, users_data
    if trans_data is None or users_data is None:
        trans_data, users_data = load_data_components()
    return trans_data, users_data

# Endpoint to match users based on transaction ID
@app.post("/match-users/", response_model=Dict[str, Union[List[Dict[str, Union[str, float]]], int]], tags=["User Matching"])
def match_users(transaction_id: str, data: Tuple[pd.DataFrame, pd.DataFrame] = Depends(get_loaded_data)):
    try:
        trans_data, users_data = data
        
        transaction_row = trans_data[trans_data['id'] == transaction_id]
        if transaction_row.empty:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        transaction_description = transaction_row.iloc[0]['description']
        transaction_user_name = extract_name(transaction_description)

        if transaction_user_name == '':
            transaction_user_name = extract_name_spacy(transaction_description)
        
        if transaction_user_name == '':
            transaction_user_name = extract_name_distilbert(transaction_description)
        
        if not transaction_user_name:
            raise HTTPException(status_code=404, detail="User name not found in transaction description")
        
        matches = []
        for _, user_row in users_data.iterrows():
            user_name = user_row['name']
            if isinstance(transaction_user_name, str) and isinstance(user_name, str):
                match_metric = fuzz.ratio(transaction_user_name, user_row['name'])
                matches.append({"id": user_row['id'], "match_metric": match_metric})
        
        matches.sort(key=lambda x: x["match_metric"], reverse=True)
        
        result = {
            "users": matches,
            "total_number_of_matches": len(matches)
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# Endpoint to find similar transactions based on input string
@app.post("/similar-transactions/", response_model=Dict[str, Union[List[Dict[str, Union[str, float]]], int]], tags=["Similar Transactions"])
def find_similar_transactions(input_string: str, data: Tuple[pd.DataFrame, pd.DataFrame] = Depends(get_loaded_data)):
    try:
        trans_data, _ = data
        
        input_tokens = tokenizer(input_string, return_tensors='pt')
        with torch.no_grad():
            input_output = model(**input_tokens)
            input_embedding = input_output.last_hidden_state.mean(dim=1).detach().numpy()
        
        transaction_tokens = tokenizer(list(trans_data['description']), padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            transaction_outputs = model(**transaction_tokens)
            transaction_embeddings = transaction_outputs.last_hidden_state.mean(dim=1).detach().numpy()
        
        similarity_scores = cosine_similarity(input_embedding, transaction_embeddings).flatten()
        trans_data['similarity_score'] = similarity_scores
        
        trans_data = trans_data.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)
        
        output = {
            'transactions': trans_data[['id', 'description']].to_dict(orient='records'),
            'total_number_of_tokens_used': len(input_tokens['input_ids'][0])
        }
        
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# Function to extract name from transaction description
def extract_name(description: str) -> str:
    try:
        description_use = description.lower()
        if "from " in description_use and " for deel" in description_use:
            start = description_use.find("from ") + len("from ")
            end = description_use.find(" for deel")
            return description[start:end]
        return ""
    except Exception as e:
        raise RuntimeError(f"Failed to extract name from description: {e}")

# Function to extract name using spaCy
def extract_name_spacy(description: str) -> str:
    try:
        doc = nlp(description)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return ""
    except Exception as e:
        raise RuntimeError(f"Failed to extract name using spaCy: {e}")

# Function to extract name using DistilBERT for NER
def extract_name_distilbert(description: str) -> str:
    try:
        ner_results = ner_pipeline(description)
        names = [result['word'] for result in ner_results if result['entity'] == 'B-PER' or result['entity'] == 'I-PER']
        return " ".join(names)
    except Exception as e:
        raise RuntimeError(f"Failed to extract name using DistilBERT: {e}")

# Run the FastAPI application with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

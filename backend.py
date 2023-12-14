import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding
from transformers import GPT2TokenizerFast
from tqdm.auto import tqdm
import os

tqdm.pandas()

import spacy
# import numpy as np

# Load spaCy model with GloVe embeddings
import en_core_web_sm

nlp = en_core_web_sm.load()

def custom_embedding(text, model_name="text-embedding-ada-002"):
    # Process the text with spaCy
    doc = nlp(text)

    # Extract word embeddings and average them to get the text embedding
    word_embeddings = [token.vector for token in doc if token.has_vector]
    
    if not word_embeddings:
        return None  # No embeddings found for any word in the text

    text_embedding = np.mean(word_embeddings, axis=0)

    # Create a response dictionary
    response = {
        "data": [
            {
                "embedding": text_embedding.tolist(),
                "index": 0,
                "object": "embedding"
            }
        ],
        "model": model_name,
        "object": "list",
        "usage": {
            "prompt_tokens": len(text.split()),
            "total_tokens": len(text.split())
        }
    }

    return response

# Example usage
text = "Rome"
response = custom_embedding(text)

if response["data"][0]["embedding"] is not None:
    print(f"Custom Embedding for '{text}': {response['data'][0]['embedding']}")
else:
    print(f"No embeddings found for words in '{text}'.")

print(response)


# import spacy
# import numpy as np

# Load spaCy model with GloVe embeddings
# import en_core_web_sm

nlp = en_core_web_sm.load()

def custom_embedding(text_list, model_name="text-embedding-ada-002"):
    embeddings = []

    for text in text_list:
        # Process the text with spaCy
        doc = nlp(text)

        # Extract word embeddings and average them to get the text embedding
        word_embeddings = [token.vector for token in doc if token.has_vector]
        
        if not word_embeddings:
            embeddings.append(None)  # No embeddings found for any word in the text
        else:
            text_embedding = np.mean(word_embeddings, axis=0)
            embeddings.append(text_embedding.tolist())

    # Create a response dictionary
    response = {
        "data": [
            {
                "embedding": emb,
                "index": idx,
                "object": "embedding"
            }
            for idx, emb in enumerate(embeddings)
        ],
        "model": model_name,
        "object": "list",
        "usage": {
            "prompt_tokens": sum(len(text.split()) for text in text_list),
            "total_tokens": sum(len(text.split()) for text in text_list)
        }
    }

    return response

# Example usage
text = ["She is running", "Fitness is good", "I am hungry", "Basketball is healthy"]
response = custom_embedding(text)

for idx, embedding in enumerate(response["data"]):
    if embedding["embedding"] is not None:
        print(f"Custom Embedding for '{text[idx]}': {embedding['embedding']}")
    else:
        print(f"No embeddings found for words in '{text[idx]}'.")

print(response)

emb1 = response['data'][0]['embedding']
emb2 = response['data'][1]['embedding']
emb3 = response['data'][2]['embedding']
emb4 = response['data'][3]['embedding']

np.dot(emb1, emb2)
np.dot(emb2, emb4)

df = pd.read_csv('/content/Dronealexa.csv')
df = df.dropna()
df.info()
df.head()
df['combined'] = "Title: " + df['Title'].str.strip() + "; URL: " + df['URL'].str.strip() + "; Publication Year: " + df['Publication Year'].astype(str).str.strip() + "; Abstract: " + df['Abstract'].str.strip()
df.head()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

df['n_tokens'] = df.combined.progress_apply(lambda x: len(tokenizer.encode(x)))
df = df[df.n_tokens < 8000]
df.info()
df.head()


# import spacy
# import numpy as np

# Load spaCy model with GloVe embeddings
# import en_core_web_sm

nlp = en_core_web_sm.load()

def get_embeddings(text, model):
    # Process the text with spaCy
    doc = model(text)

    # Extract word embeddings and average them to get the text embedding
    word_embeddings = [token.vector for token in doc if token.has_vector]
    
    if not word_embeddings:
        return None  # No embeddings found for any word in the text

    text_embedding = np.mean(word_embeddings, axis=0)

    # Create a response dictionary
    response = {
        "data": [
            {
                "embedding": text_embedding.tolist(),
                "index": 0,
                "object": "embedding"
            }
        ],
        "model": model.meta["name"],
        "object": "list",
        "usage": {
            "prompt_tokens": len(text.split()),
            "total_tokens": len(doc)
        }
    }

    return response

# Example usage
input_text = "Your input text goes here"
custom_model = nlp  # You can replace this with any other spaCy model

# Renaming 'input_text' to avoid conflict with the built-in 'input' function
text_to_process = input_text  

response = get_embeddings(text_to_process, custom_model)

if response["data"][0]["embedding"] is not None:
    print(f"Custom Embedding for '{text_to_process}': {response['data'][0]['embedding']}")
else:
    print(f"No embeddings found for words in '{text_to_process}'.")

print(response)

from tqdm import tqdm

batch_size = 2000
model_name = 'text-embedding-ada-002'

# Assuming df is your DataFrame
for i in tqdm(range(0, len(df.combined), batch_size)):
    # find end of batch
    i_end = min(i + batch_size, len(df.combined))
    
    # Get embeddings for the current batch
    batch_text = list(df.combined)[i:i_end]
    
    # Initialize an empty list to store the embeddings for each text in the batch
    batch_embeddings = []
    
    # Process each text in the batch and get embeddings
    for text in batch_text:
        response = get_embeddings(text, nlp)
        
        # Check if embeddings were found
        if response and response["data"][0]["embedding"] is not None:
            batch_embeddings.append(response["data"][0]["embedding"])
        else:
            # Handle the case where no embeddings are found for a text
            batch_embeddings.append(None)

    # Update the DataFrame with the embeddings
    for j in range(i, i_end):
        df.loc[j, 'ada_vector'] = str(batch_embeddings[j - i])
        
df.head()
df.info()
df['ada_vector'] = df.ada_vector.progress_apply(eval).progress_apply(np.array)
df.to_csv('embeddings_chatbot.csv',index=False)
df=pd.read_csv('embeddings_chatbot.csv')

user_query = input("Enter query - ")

query_response = get_embeddings(user_query, nlp)

if query_response["data"][0]["embedding"] is not None:
    print(f"Embedding for '{user_query}': {query_response['data'][0]['embedding']}")
else:
    print(f"No embeddings found for words in '{user_query}'.")
    
searchvector = get_embeddings(user_query, custom_model)["data"][0]["embedding"]



from sklearn.metrics.pairwise import cosine_similarity

# Assuming df['ada_vector'] contains the vectors you want to compare

# Ensure 'ada_vector' column contains valid numeric arrays
df['ada_vector'] = df['ada_vector'].apply(lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) else x)

# Filter out rows where 'ada_vector' is not a valid numeric array
valid_rows = df['ada_vector'].apply(lambda x: isinstance(x, np.ndarray))

# Calculate cosine similarity only for valid rows
df.loc[valid_rows, 'similarities'] = df.loc[valid_rows, 'ada_vector'].apply(
    lambda x: cosine_similarity([x], [searchvector])[0][0]
)

# If you are using the 'progress_apply' from the 'tqdm' library
# You can keep it as follows:
# df.loc[valid_rows, 'similarities'] = df.loc[valid_rows, 'ada_vector'].progress_apply(
#     lambda x: cosine_similarity([x], [searchvector])[0][0]
# )

df.head()
df.sort_values('similarities', ascending = False)
result = df.sort_values('similarities', ascending = False).head(3)

result.head()

xc = list(result.combined)

def construct_prompt(query, xc):
  context = ''
  for i in range(3):
    context += xc[i] + "\n"
  header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
  header += context + "\n\n Q: " + query + "\n A:"
  return header



from transformers import pipeline

summarizer = pipeline("summarization")
Fresult = construct_prompt(user_query, xc)
summarizer("\n".join(xc), max_length=130, min_length=30, do_sample=False)
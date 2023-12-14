import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
from transformers import pipeline

# Load the CSV file and preprocess it
def load_csv_and_preprocess(csv_file):
    df = pd.read_csv(csv_file)
    df = df.dropna().head(100000)
    
    column_names = list(df.columns)
    df['combined'] = df.apply(lambda x: "Title: " + '; '.join(x[column_names].astype(str)), axis=1)
    df['combined'] = df['combined'].str.strip()

    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(df['combined'])
    
    return df, vectorizer, embeddings

# Initialize the summarization pipeline outside of the chatbot_response function
summarizer = pipeline("summarization")

# Perform semantic search for a given query
def semantic_search(df, vectorizer, embeddings, query):
    search_vector = vectorizer.transform([query])
    similarities = cosine_similarity(search_vector, embeddings).flatten()
    df['similarities'] = similarities
    result = df.sort_values('similarities', ascending=False).head(3)
    
    return result['combined'].tolist()

# Define the chatbot response function with summarization
def chatbot_response(query, history):
    if not query.strip():
        return "", history
    search_results = semantic_search(df, vectorizer, embeddings, query)
    # Summarize the search results
    summary = summarizer("\n".join(search_results), max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    # Format the summarized response and update the chat history
    history = f"{history}User: {query}\nBot: {summary}\n\n"
    return "", history  # Clear the input box after each message, update history

# Load CSV and preprocess on server startup
csv_file_path = "Dronealexa.csv"  # Update this to your CSV file path
df, vectorizer, embeddings = load_csv_and_preprocess(csv_file_path)

# Define a function to handle feedback
def handle_feedback(feedback, response, history_box):
    # Simple logic to prepend feedback to the user's query
    # This could be replaced with more sophisticated logic or ML model updating
    response = f"Based on your feedback ('{feedback}'): {response}"
    history = history_box + "\nBot: " + response + "\n"
    return "", history  # Update the history with the feedback-aware response


# Gradio Blocks Interface
with gr.Blocks() as blocks_app:
    gr.Markdown("<h1 style='text-align: center;'>Explore Science & Technology with Chatbot</h1>")
    history_box = gr.Textbox(label="", value="", interactive=False, lines=20)
    with gr.Row():
        query_input = gr.Textbox(show_label=False, placeholder="Type your message here...", lines=1)
    with gr.Row():
        send_button = gr.Button("Send")

    send_button.click(
        fn=chatbot_response,
        inputs=[query_input, history_box],
        outputs=[query_input, history_box]
    )
    feedback_input = gr.Textbox(show_label=False, placeholder="Type your feedback here...", lines=1)
    feedback_button = gr.Button("Submit Feedback")

    feedback_button.click(
        fn=handle_feedback,
        inputs=[feedback_input, history_box, history_box],
        outputs=[query_input, history_box]
    )
blocks_app.launch(share=True)
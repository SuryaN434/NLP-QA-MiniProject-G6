from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import pandas as pd
import os
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:5001"])  # Enable CORS for frontend communication

# Load BERT model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name, trust_remote_code=True)

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "qa_dataset_final_cleaned.csv")
df = pd.read_csv(DATASET_PATH)

# Ensure column names are correct
df.columns = df.columns.str.lower()  # Convert column names to lowercase
if "question" not in df.columns or "answer" not in df.columns or "category" not in df.columns:
    raise KeyError("Dataset must contain 'question', 'answer', and 'category' columns")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["question"])  # Convert dataset questions into vectors

# USE THIS VERSION IF WE WANT THE FULL ANSWER FROM THE DATASETS
def get_best_context(user_question, topic):
    """Finds the most relevant answer using TF-IDF cosine similarity and ensures full retrieval."""
    filtered_df = df[df["category"].str.lower() == topic.lower()]
    
    if filtered_df.empty:
        return None

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(filtered_df["question"])

    user_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    THRESHOLD = 0.8  # Adjust if necessary
    if best_score < THRESHOLD:
        return None

    # Return full answer from dataset
    best_answer = filtered_df.iloc[best_index]["answer"]
    return best_answer.strip()  # Ensure no unwanted 


# USE THIS VERSION IF WE WANT TO EXPECT MORE SUMMARIZED RESULT
# def get_best_context(user_question, topic):
#     """Find the most relevant question-answer pair using TF-IDF cosine similarity within the selected category."""
#     filtered_df = df[df["category"].str.lower() == topic.lower()]  # Filter dataset by topic

#     if filtered_df.empty:  # If no matching topic is found, return None
#         return None

#     # Vectorize only the questions from the filtered dataset
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(filtered_df["question"])

#     # Transform user question
#     user_vector = vectorizer.transform([user_question])
#     similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

#     # Find the highest similarity score
#     best_index = similarities.argmax()
#     best_score = similarities[best_index]

#     THRESHOLD = 0.7  # Set a threshold for similarity (adjust as needed)
#     return filtered_df.iloc[best_index]["answer"] if best_score > THRESHOLD else None


@app.route("/ask", methods=["POST"])
def answer_question():
    data = request.get_json()
    question = data.get("question")
    topic = data.get("topic")

    if not question or not topic:
        return jsonify({"error": "Question and topic are required"}), 400

    # Retrieve the most relevant answer using TF-IDF
    context = get_best_context(question, topic)

    if not context:
        return jsonify({"answer": "Sorry, I couldn't find relevant information 🥲."})

    # Use BERT for more precise answer extraction
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    # Convert token IDs to words correctly
    answer_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end])

    # Properly join words while ensuring correct formatting
    extracted_answer = " ".join(answer_tokens)

    # **Fix token artifacts:**
    extracted_answer = extracted_answer.replace(" ##", "")  # Remove BERT subword markers
    extracted_answer = extracted_answer.replace("undefined", "").strip()  # Remove "undefined" issue
    extracted_answer = extracted_answer.replace(" ,", ",").replace(" .", ".")  # Fix spacing

    # If extracted answer is empty, use the full TF-IDF context
    final_answer = extracted_answer if extracted_answer and extracted_answer.strip() else context

    # use this if we want full answer
    return jsonify({"answer": context})

    # use this if we want a more summarized answer
    # return jsonify({"answer": final_answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

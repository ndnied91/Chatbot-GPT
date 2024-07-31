import csv
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()  # take environment variables from .env.

client = OpenAI(api_key=os.getenv('OPENAI_APIKEY'))

def normalize_text(text):
    return text.strip().lower()

# Function to load QA data from a CSV file and store it in a dictionary
def load_qa_dict_from_csv(file_path):
    qa_dict = {}
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row if present
        for row in reader:
            if len(row) == 2:
                question, answer = row
                qa_dict[normalize_text(question)] = answer
    return qa_dict

# Load the QA data from the CSV file
qa_dict = load_qa_dict_from_csv('data.csv')

# Load the pre-trained model for semantic search
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Prepare dataset embeddings for semantic search
questions = list(qa_dict.keys())
question_embeddings = model.encode(questions, convert_to_tensor=True)

conversation_history = [{"role": "system", "content": "You are a helpful assistant to Daniel, he is a software engineer at Microsoft. If you get asked any specific questions, please refer to the data that is provided. Your name is Reginald Pembroke, you're a sophisticated assistant."}]

def get_answer_from_chatgpt(question):
    conversation_history.append({"role": "user", "content": question})
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation_history
    )
    # Correct access to the message content
    answer = response.choices[0].message.content.strip()
    return answer


# Function to get an answer from the dataset using exact matching and semantic search
def get_answer_from_dataset(question, qa_dict, model, question_embeddings):
    normalized_question = normalize_text(question)
    
    # Exact match
    exact_match = qa_dict.get(normalized_question)
    if exact_match:
        return exact_match
    
    # Semantic search
    question_embedding = model.encode(normalized_question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_embedding, question_embeddings)[0]
    best_match_idx = scores.argmax().item()
    
    if scores[best_match_idx] > 0.7:  # Adjust threshold as needed
        best_match_question = questions[best_match_idx]
        return qa_dict[best_match_question]
    
    if any(keyword in normalized_question for keyword in ["daniel", "linkedin", "background"]):
        # Fallback message with LinkedIn URL
        return f"Hmm, I'm not too sure about this answer, but please visit Daniel's LinkedIn profile for more information: https://www.linkedin.com/in/daniel-niedzwiedzki"
    
    # Fallback to ChatGPT
    return get_answer_from_chatgpt(question)

# Main chatbot function to be imported
def chatbot(question):
    return get_answer_from_dataset(question, qa_dict, model, question_embeddings)

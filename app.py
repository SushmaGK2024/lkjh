# Import necessary libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import sys
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import torch
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load the fine-tuned model
model_path = "fine_tuned_bert_model"
model = BertForSequenceClassification.from_pretrained(model_path)

# Specify the path to the label encoder file
label_encoder_path = "label_encoder.pkl"

# Load the label encoder
label_encoder = joblib.load(label_encoder_path)

# Preprocess the text
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters, numbers, and punctuations
    return text

# Tokenize and lemmatize the text
def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words('english'))  # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()  # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Predict category of the question
def predict_category(question):
    cleaned_text = preprocess_text(question)  # Clean and preprocess the question
    tokens = tokenize_and_lemmatize(cleaned_text)  # Tokenize and lemmatize
    encoded_dict = tokenizer.encode_plus(
                        " ".join(tokens),
                        add_special_tokens=True,
                        max_length=128,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                   )  # Tokenize the question using BERT tokenizer
    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    predicted_label = label_encoder.inverse_transform(logits.argmax(axis=1).detach().numpy())[0]  # Decode predicted label
    return predicted_label

# Route to handle POST request for adding experience
@app.route('/add-experience', methods=['POST'])
def add_experience():
    # Get data from request
    data = request.json
    overall_experience = data.get('overallExperience')
    # Extract questions from overall experience
    sentences = sent_tokenize(overall_experience)
    lemmatized_words = tokenize_and_lemmatize(overall_experience)
    extracted_questions = extract_questions(sentences, lemmatized_words)
    # Predict category for each question
    categories = [predict_category(question) for question in extracted_questions]
    # Store experience and questions in database
    # Example: code to store data in database goes here
    return jsonify({'message': 'Experience added successfully'})

# Route to handle extract questions request
@app.route('/extract-questions', methods=['POST'])
def extract_questions_route():
    data = request.json
    overall_experience = data.get('overallExperience')
    # Extract questions from overall experience
    text=overall_experience
    sentences = sent_tokenize(text)

    # Tokenize words in each sentence
    words = [word_tokenize(sentence) for sentence in sentences]

    # Flatten the list of words
    words = [word for sentence_words in words for word in sentence_words]

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words("english"))
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    extracted_questions = extract_questions(sentences, lemmatized_words)
    return jsonify({'questions': extracted_questions})

# Function to extract questions from sentences
def extract_questions(sentences, lemmatized_words):
    print("sentences : ", sentences)
    print("lem : ", lemmatized_words)
    questions = []
    interrogative_words = ["who", "what", "where", "when", "why", "how", "which", "whose", "whom", "explain", "tell", "do", "did", "question", "questions", "find", "write", "code", "programming", "approach"]
    try:
        for sentence in sentences:
            words = word_tokenize(sentence)  # Tokenize the sentence into words
            if any(word.lower() in interrogative_words or word.endswith('?') for word in words):
                questions.append(sentence)
        return questions
    except Exception as e:
        print(f"Error during question extraction: {e}", file=sys.stderr)
        return None
@app.route('/predict-category', methods=['POST'])
def predict_category_route():
    # Get the question from the request body
    question = request.json.get('question')

    # Predict the category
    predicted_category = predict_category(question)
    print(predicted_category,question)
    # Return the predicted category as a JSON response
    return jsonify({'predicted_category': predicted_category})

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app in debug mode

import os
import re
import nltk
import torch
import joblib
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load the fine-tuned model
model_path = "thanuja2510/pymodel"
model = BertForSequenceClassification.from_pretrained(model_path)

# Specify the path to the label encoder file
label_encoder_path = "label_encoder.pkl"

# Load the label encoder
label_encoder = joblib.load(label_encoder_path)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

def predict_category(question):
    cleaned_text = preprocess_text(question)
    tokens = tokenize_and_lemmatize(cleaned_text)
    encoded_dict = tokenizer.encode_plus(
                        " ".join(tokens),
                        add_special_tokens=True,
                        max_length=128,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                   )
    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    predicted_label = label_encoder.inverse_transform(logits.argmax(axis=1).detach().numpy())[0]
    return predicted_label

@app.route('/add-experience', methods=['POST'])
def add_experience():
    data = request.json
    overall_experience = data.get('overallExperience')
    sentences = sent_tokenize(overall_experience)
    lemmatized_words = tokenize_and_lemmatize(overall_experience)
    extracted_questions = extract_questions(sentences, lemmatized_words)
    categories = [predict_category(question) for question in extracted_questions]
    return jsonify({'message': 'Experience added successfully'})

@app.route('/extract-questions', methods=['POST'])
def extract_questions_route():
    data = request.json
    overall_experience = data.get('overallExperience')
    text = overall_experience
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]
    words = [word for sentence_words in words for word in sentence_words]
    stop_words = set(stopwords.words("english"))
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    extracted_questions = extract_questions(sentences, lemmatized_words)
    return jsonify({'questions': extracted_questions})

def extract_questions(sentences, lemmatized_words):
    questions = []
    interrogative_words = ["who", "what", "where", "when", "why", "how", "which", "whose", "whom", "explain", "tell", "do", "did", "question", "questions", "find", "write", "code", "programming", "approach"]
    for sentence in sentences:
        words = word_tokenize(sentence)
        if any(word.lower() in interrogative_words or word.endswith('?') for word in words):
            questions.append(sentence)
    return questions

@app.route('/predict-category', methods=['POST'])
def predict_category_route():
    question = request.json.get('question')
    predicted_category = predict_category(question)
    return jsonify({'predicted_category': predicted_category})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

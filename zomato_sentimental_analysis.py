import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load and preprocess dataset
zomato_df = pd.read_csv('zomato.csv', encoding='latin1')
zomato_df = zomato_df[['reviews_list', 'rate']].dropna()
zomato_df = zomato_df[zomato_df['rate'].str.match(r'^\d+(\.\d+)?/5$', na=False)]
zomato_df['rate'] = zomato_df['rate'].str.replace('/5', '').astype(float)

# Assign sentiment labels
def map_sentiment(rate):
    if rate <= 2.5:
        return 'Negative'
    elif 2.5 < rate <= 3.5:
        return 'Neutral'
    else:
        return 'Positive'

zomato_df['sentiment'] = zomato_df['rate'].apply(map_sentiment)

# Clean reviews
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

zomato_df['reviews_list'] = zomato_df['reviews_list'].apply(clean_text)

# Train-test split
X = zomato_df['reviews_list']
y = zomato_df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Model training and evaluation
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "SVM": SVC(class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    print(f"{name} - Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Hugging Face BERT Sentiment Analysis
label_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
zomato_df['label'] = zomato_df['sentiment'].map(label_mapping)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    zomato_df['reviews_list'], zomato_df['label'], test_size=0.2, random_state=42, stratify=zomato_df['label']
)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({'text': train_texts.tolist(), 'label': train_labels.tolist()})
test_dataset = Dataset.from_dict({'text': test_texts.tolist(), 'label': test_labels.tolist()})

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
train_tokenized = train_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)

# Define model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    tokenizer=tokenizer
)

# Train and evaluate
trainer.train()
results = trainer.evaluate()

# Predictions
predictions = trainer.predict(test_tokenized)
predicted_labels = predictions.predictions.argmax(axis=1)

reverse_label_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
predicted_sentiments = [reverse_label_mapping[label] for label in predicted_labels]

# Evaluate
print(classification_report([reverse_label_mapping[label] for label in test_labels], predicted_sentiments))

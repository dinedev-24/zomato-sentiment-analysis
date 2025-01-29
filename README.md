# Zomato Sentiment Analysis

This project performs sentiment analysis on restaurant reviews from Zomato, combining both general sentiment classification and aspect-based sentiment analysis.

## Overview

The goal of this project is to gain insights into customer sentiments and identify trends related to different aspects of the dining experience, such as food quality, service, and ambiance. By applying natural language processing (NLP) techniques and state-of-the-art machine learning models, the project provides a structured way to measure customer satisfaction and pinpoint areas for improvement.

## Features

- **General Sentiment Analysis**: Classify reviews into Positive, Neutral, or Negative sentiments.
- **Aspect-Based Sentiment Analysis (ABSA)**: Drill down into specific aspects (food, service, place) and identify corresponding sentiment trends.
- **Data Preprocessing**: Clean and prepare textual data by removing noise, handling missing values, and formatting ratings.
- **Visualization**: Generate clear, informative visualizations to show sentiment distributions and aspect trends.
- **Model Comparison**: Evaluate multiple models including Logistic Regression, Random Forest, SVM, XGBoost, and a fine-tuned BERT model.

## Technologies Used

- **Data Handling**: pandas, numpy
- **Data Visualization**: matplotlib, seaborn, Streamlit (for the dashboard interface)
- **Machine Learning Models**: Logistic Regression, SVM, Random Forest, XGBoost, BERT (from Hugging Face Transformers)
- **Natural Language Processing (NLP)**: TF-IDF vectorization, BERT tokenization, aspect-based sentiment extraction
- **Development Tools**: Python 3.9, Jupyter Notebook, Streamlit, Git

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dinedev-24/zomato-sentiment-analysis.git
   cd zomato-sentiment-analysis

2. **Set up your environment:**

Ensure Python 3.9 is installed.
Install required packages using requirements.txt:
pip install -r requirements.txt

3. **Run the application:**

Use the provided Streamlit app for interactive visualization:
streamlit run zomatoapp.py

4. **Explore the code:**

zomato_sentimental_analysis.py: The main script for data processing, sentiment analysis, and model comparison.
updated_sentiment_data.csv: The final dataset with aspect-specific and general sentiment annotations.
zomatoapp.py: The Streamlit-based dashboard.

**Example Visualizations**
  [Insert screenshots or describe key visualizations here. For example:]
  
  General sentiment distribution bar charts.
  Line charts showing trends over time for aspect-based sentiment.
  Comparison of model performances using accuracy and F1-score metrics.
**Results**
**General Sentiment Analysis:**
  Overall accuracy: XX%
  Most reviews fall into the Neutral sentiment category.
**Aspect-Based Sentiment Analysis:**
  Food: Predominantly Positive
  Service: Balanced between Neutral and Positive
  Place: Mostly Neutral
**Future Improvements**
  Expand aspect analysis to include additional aspects like pricing, ambiance, and customer service.
  Integrate a real-time review stream for dynamic sentiment updates.
  Explore additional NLP models (e.g., RoBERTa, DistilBERT) for performance improvements.

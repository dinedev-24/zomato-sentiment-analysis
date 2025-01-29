                        **Zomato Sentiment Analysis Project**
**Overview**
This project performs sentiment analysis on Zomato customer reviews using a combination of traditional machine learning models and BERT-based transformers. The dataset contains review text and ratings, and the analysis includes both general sentiment classification and aspect-based sentiment analysis (ABSA).

**Key Features**
**General Sentiment Analysis:**
Assigns overall sentiment (Positive, Neutral, Negative) to each review using ratings and review text.
**Aspect-Based Sentiment Analysis (ABSA):**
Provides sentiment insights on specific aspects such as food, service, and ambiance.
**Machine Learning Models:**
Includes Logistic Regression, SVM, Random Forest, and XGBoost for baseline comparison.
**BERT Integration:**
Fine-tuned BERT model for more accurate sentiment classification.
**Visualizations:**
Displays sentiment distributions and compares performance metrics across models.
**Project Structure**
Requirements.txt:
Contains the list of Python dependencies required for the project.
**updated_sentiment_data.csv**:
The preprocessed dataset including review texts, ratings, and sentiment labels.
**zomato_sentimental_analysis.py**:
The main script for performing sentiment analysis using both traditional models and BERT.
**zomatoapp.py**:
A Streamlit app that provides an interactive dashboard for visualizing sentiment distributions.
**How to Run**
**Setup Environment**:
Make sure you have Python installed. Install the required packages by running:
pip install -r Requirements.txt
**Run the Sentiment Analysis Script:**
To perform sentiment analysis and compare model performance, run:
python zomato_sentimental_analysis.py
**Start the Streamlit App:**
To view the visualizations and sentiment distribution dashboard, run:
streamlit run zomatoapp.py
Then open the local URL provided by Streamlit in your browser.
**Visualizations**
Sentiment Distributions:
Visualizes overall sentiment and aspect-based sentiment across the dataset.
**Model Performance Comparison:**
Bar charts comparing accuracy and F1-scores for each model, including BERT.
**Future Work**
Enhancing the BERT model by experimenting with different architectures like RoBERTa or DistilBERT.
Exploring additional aspects (e.g., price, ambiance) for more granular ABSA.
Integrating the solution into a live dashboard for real-time review monitoring.

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data_file = 'updated_sentiment_data.csv'  # Make sure this is the correct path on EC2

# Read the CSV file
df = pd.read_csv(data_file)

# Assign a starting date and create a sequence of dates
start_date = pd.Timestamp('2024-01-01')
df['date_column'] = [start_date + pd.Timedelta(days=i) for i in range(len(df))]

# Ensure the date column is set as datetime64[ns] and check the first few rows
df['date_column'] = pd.to_datetime(df['date_column'], errors='coerce')
print("date_column dtype:", df['date_column'].dtype)
print("Sample dates from date_column:")
print(df['date_column'].head())

# Display the first few rows to understand the data structure
st.title("Sentiment Analysis Dashboard")
st.write("Loaded Data:")
st.write(df.head())

# Sentiment Analysis - General Sentiment Distribution
st.header("General Sentiment Distribution")
general_sentiment = df['general_sentiment'].value_counts()  # Adjusted the column name
st.bar_chart(general_sentiment)

# Negative Sentiment Trends
st.header("Negative Sentiment Trends")
negative_subset = df[df['general_sentiment'] == 'Negative']
negative_trends = negative_subset.set_index('date_column').resample('W').size()
st.line_chart(negative_trends)

# Aspect-wise Sentiment Distribution (Food, Service, Place)
st.header("Aspect-wise Sentiment Distribution")
fig, ax = plt.subplots(3, 1, figsize=(8, 12))

# Food Sentiment
food_sentiment = df['food_sentiment'].value_counts()
ax[0].bar(food_sentiment.index, food_sentiment.values, color='green')
ax[0].set_title('Food Sentiment Distribution')
ax[0].set_xlabel('Sentiment')
ax[0].set_ylabel('Frequency')
st.header("Food Sentiment Trends")
food_negative_subset = df[df['food_sentiment'] == 'Negative']
food_negative_trends = food_negative_subset.set_index('date_column').resample('W').size()
st.line_chart(food_negative_trends)

# Service Sentiment
service_sentiment = df['service_sentiment'].value_counts()
ax[1].bar(service_sentiment.index, service_sentiment.values, color='blue')
ax[1].set_title('Service Sentiment Distribution')
ax[1].set_xlabel('Sentiment')
ax[1].set_ylabel('Frequency')
st.header("Service Sentiment Trends")
service_negative_subset = df[df['service_sentiment'] == 'Negative']
service_negative_trends = service_negative_subset.set_index('date_column').resample('W').size()
st.line_chart(service_negative_trends)

# Place Sentiment
place_sentiment = df['place_sentiment'].value_counts()
ax[2].bar(place_sentiment.index, place_sentiment.values, color='red')
ax[2].set_title('Place Sentiment Distribution')
ax[2].set_xlabel('Sentiment')
ax[2].set_ylabel('Frequency')

st.pyplot(fig)

# Show the raw count data for further insight
st.write("Sentiment Counts:")
st.write(df[['food_sentiment', 'service_sentiment', 'place_sentiment']].apply(pd.Series.value_counts).fillna(0))

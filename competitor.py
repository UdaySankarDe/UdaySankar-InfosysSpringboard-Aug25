import pandas as pd
from transformers import pipeline

# Load reviews
reviews_df = pd.read_csv("reviews.csv")

# Sentiment model
sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Function to get sentiment
def get_sentiment(text):
    try:
        result = sentiment_model(text[:512])[0]
        label_map = {"LABEL_0":"Negative","LABEL_1":"Neutral","LABEL_2":"Positive"}
        return label_map.get(result['label'],"Neutral"), result['score']
    except:
        return "Neutral", 0.0

# Apply sentiment
reviews_df[['Sentiment','Sentiment_Score']] = reviews_df['Review_Text'].apply(lambda x: pd.Series(get_sentiment(str(x))))

# Save
reviews_df.to_csv("reviews_with_sentiment.csv", index=False)
print("Sentiment analysis completed. Saved to reviews_with_sentiment.csv")

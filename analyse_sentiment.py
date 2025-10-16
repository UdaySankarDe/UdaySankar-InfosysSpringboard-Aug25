import pandas as pd
import plotly.express as px
import os


# Step 1: Load reviews_with_sentiment.csv

file_path = r"C:\Users\Uday Sankar De\Desktop\Springboard intern 2025\reviews_with_sentiment.csv"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit()

reviews = pd.read_csv(file_path)


required_cols = ['Product_ASIN', 'Review_Text', 'Sentiment', 'Sentiment_Score', 'Review_Date']
for col in required_cols:
    if col not in reviews.columns:
        print(f"Missing column in CSV: {col}")
        exit()


reviews['Review_Date'] = pd.to_datetime(reviews['Review_Date'], errors='coerce')
reviews.dropna(subset=['Review_Date'], inplace=True)


# Step 2: Aggregate sentiment per product (weekly)

sentiment_trends = reviews.groupby(['Product_ASIN', pd.Grouper(key='Review_Date', freq='W')]) \
                          .agg(avg_sentiment=('Sentiment_Score', 'mean'),
                               review_count=('Sentiment_Score', 'count')) \
                          .reset_index()

sentiment_trends.to_csv("sentiment_trends.csv", index=False)
print("✅ Sentiment trends saved to sentiment_trends.csv")

# Step 3: Visualize sentiment trends per product

for asin in sentiment_trends['Product_ASIN'].unique():
    df = sentiment_trends[sentiment_trends['Product_ASIN'] == asin]
    fig = px.line(df, x='Review_Date', y='avg_sentiment',
                  title=f'Sentiment Trend for {asin}',
                  markers=True)
    fig.show()


# Step 4: Top positive & negative reviews

top_positive = reviews.sort_values('Sentiment_Score', ascending=False).head(10)
top_negative = reviews.sort_values('Sentiment_Score', ascending=True).head(10)

top_positive.to_csv("top_positive_reviews.csv", index=False)
top_negative.to_csv("top_negative_reviews.csv", index=False)

print("✅ Top positive reviews saved to top_positive_reviews.csv")
print("✅ Top negative reviews saved to top_negative_reviews.csv")


# Step 5: Summary for strategy

summary = []

for asin in reviews['Product_ASIN'].unique():
    prod_reviews = reviews[reviews['Product_ASIN'] == asin]
    avg_sentiment = prod_reviews['Sentiment_Score'].mean()
    review_count = prod_reviews.shape[0]
    
    summary.append({
        'Product_ASIN': asin,
        'Average_Sentiment': avg_sentiment,
        'Review_Count': review_count
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("sentiment_summary.csv", index=False)
print("✅ Sentiment summary saved to sentiment_summary.csv")

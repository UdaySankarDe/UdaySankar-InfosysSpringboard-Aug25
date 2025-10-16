import pandas as pd
import streamlit as st
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

##############################
# File Paths
##############################
REVIEWS_FILE = r"C:\Users\Uday Sankar De\Desktop\Springboard intern 2025\reviews_with_sentiment.csv"
PRODUCTS_FILE = r"C:\Users\Uday Sankar De\Desktop\Springboard intern 2025\Amazon\products.csv"

##############################
# Load CSVs
##############################
@st.cache_data
def load_reviews(file_path):
    df = pd.read_csv(file_path)
    df["Review_Date"] = pd.to_datetime(df["Review_Date"], errors="coerce")
    df.dropna(subset=["Review_Text", "Sentiment", "Sentiment_Score"], inplace=True)
    return df

@st.cache_data
def load_products(file_path):
    df = pd.read_csv(file_path)
    df.dropna(subset=["Discount"], inplace=True)  # Keep only rows with Discount
    df["Discount"] = df["Discount"].astype(float)
    # Do NOT reference 'Date' because your CSV doesn't have it
    return df


reviews_df = load_reviews(REVIEWS_FILE)
products_df = load_products(PRODUCTS_FILE)

##############################
# Streamlit Setup
##############################
st.set_page_config(
    page_title="Sentiment Dashboard",
    page_icon="🛒",
    layout="wide"
)

st.title("📊 Product Sentiment Dashboard")

##############################
# Product Selection
##############################
products = reviews_df["Product_Name"].unique().tolist()
selected_product = st.selectbox("Select a product to analyze", products)

product_reviews = reviews_df[reviews_df["Product_Name"] == selected_product]
product_data = products_df[products_df["Product_Name"] == selected_product]

##############################
# Show Product Image
##############################
product_asin = product_reviews["Product_ASIN"].iloc[0] if "Product_ASIN" in product_reviews.columns else None
if product_asin:
    product_image_url = f"https://images.amazon.com/images/P/{product_asin}.jpg"
    st.image(product_image_url, width=300, caption=selected_product)
else:
    st.warning("⚠️ Product image not available")

##############################
# Sentiment Summary
##############################
st.header("🗣️ Sentiment Analysis Summary")
sentiment_counts = product_reviews["Sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]

fig = px.bar(
    sentiment_counts,
    x="Sentiment",
    y="Count",
    color="Sentiment",
    color_discrete_map={
        "Positive": "#69f542",
        "Neutral": "#42ddf5",
        "Negative": "#f54248"
    },
    title=f"Sentiment Distribution for {selected_product}"
)
st.plotly_chart(fig, use_container_width=True)

avg_score = product_reviews["Sentiment_Score"].mean()
st.metric(label="Average Sentiment Score", value=f"{avg_score:.2f}")

##############################
# Discount Forecasting (No Date Column)
##############################
if not product_data.empty and "Discount" in product_data.columns:
    st.subheader("🔮 Forecasted Discounts for Next 5 Points")
    discount_series = product_data["Discount"].reset_index(drop=True)
    
    if len(discount_series) >= 6:
        model = ARIMA(discount_series, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=5)
        st.line_chart(forecast)
    else:
        st.warning("⚠️ Not enough discount data to forecast (need at least 6 points).")

##############################
# Top Reviews
##############################
st.header("⭐ Top Reviews")

top_positive = product_reviews[product_reviews["Sentiment"] == "Positive"].sort_values(
    by="Sentiment_Score", ascending=False
).head(5)

top_negative = product_reviews[product_reviews["Sentiment"] == "Negative"].sort_values(
    by="Sentiment_Score"
).head(5)

st.subheader("Top 5 Positive Reviews")
for i, row in top_positive.iterrows():
    st.markdown(f"**{row['Review_Title']}**")
    st.write(row['Review_Text'])
    st.write("---")

st.subheader("Top 5 Negative Reviews")
for i, row in top_negative.iterrows():
    st.markdown(f"**{row['Review_Title']}**")
    st.write(row['Review_Text'])
    st.write("---")

##############################
# Simple Recommendations
##############################
st.header("💡 Simple Strategy Recommendations")

positive_pct = len(top_positive) / len(product_reviews) * 100
negative_pct = len(top_negative) / len(product_reviews) * 100

if negative_pct > 50:
    st.warning("⚠️ High negative sentiment detected. Consider improving product quality or addressing common complaints.")
elif positive_pct > 60:
    st.success("✅ Positive sentiment is strong. Maintain current strategy and consider small promotions to boost sales.")
else:
    st.info("ℹ️ Mixed sentiment. Monitor reviews closely and respond to customer feedback proactively.")

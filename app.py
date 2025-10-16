import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os

# --- PATH FIX: Get Base Directory for Robust Pathing ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- File Paths ----------------
# Using os.path.join for consistency, even if the absolute paths are defined
REVIEWS_FILE = r"C:\Users\Uday Sankar De\Desktop\Springboard intern 2025\reviews_with_sentiment.csv"
PRODUCTS_FILE = r"C:\Users\Uday Sankar De\Desktop\Springboard intern 2025\Amazon\products.csv"
NOTIF_LOG = os.path.join(BASE_DIR, "My_docs", "notifications.csv")

# ---------------- Load Data ----------------
@st.cache_data
def load_reviews(file_path):
    df = pd.read_csv(file_path)
    df["Review_Date"] = pd.to_datetime(df["Review_Date"], errors="coerce")
    df.dropna(subset=["Review_Text", "Sentiment", "Sentiment_Score"], inplace=True)
    return df

@st.cache_data
def load_products(file_path):
    df = pd.read_csv(file_path)
    if df['Price'].dtype == object:
        df["Price"] = df["Price"].astype(str).str.replace('₹', '').str.replace(',', '').str.strip()
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["MRP"] = pd.to_numeric(df["MRP"], errors="coerce")
    df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce")
    return df

reviews_df = load_reviews(REVIEWS_FILE)
products_df = load_products(PRODUCTS_FILE)

# ---------------- Streamlit Page Config ----------------
st.set_page_config(
    page_title="E-Commerce Competitor Strategy Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {font-size:2.3rem;color:#1f77b4;text-align:center;margin-bottom:1rem;}
.section-header {font-size:1.6rem;color:#2e86ab;margin-top:2rem;margin-bottom:1rem;}
.positive-sentiment { color:#28a745; }
.negative-sentiment { color:#dc3545; }
.neutral-sentiment  { color:#ffc107; }
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.title("⚙ Navigation")
# --- UPDATED: Added "Recent Notifications" to the main radio buttons ---
section = st.sidebar.radio("Go to:", ["Product Analysis", "Competitor Comparison", "Strategic Recommendations", "Recent Notifications"])
products = products_df["Product_Name"].unique().tolist()
selected_product = st.sidebar.selectbox("Select Product", products)

# ---------------- Sidebar (Emptying the static notification area) ----------------
# This area is now empty as the notifications are moved to the main body
st.sidebar.markdown("---") 

# ======================================================
# ================= MAIN SECTIONS =======================
# ======================================================

# ---------------- Product Analysis ----------------
if section == "Product Analysis":
    st.markdown('<div class="main-header">Product Analysis</div>', unsafe_allow_html=True)

    # ... (Product Analysis content remains the same) ...
    if not products_df[products_df["Product_Name"] == selected_product].empty:
        prod = products_df[products_df["Product_Name"] == selected_product].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Price", f"₹{int(prod['Price'])}" if pd.notna(prod['Price']) else "N/A")
        c2.metric("MRP", f"₹{int(prod['MRP'])}" if pd.notna(prod['MRP']) else "N/A")
        c3.metric("Discount", f"{prod['Discount']:.1f}%" if pd.notna(prod['Discount']) else "N/A")

        # Product image
        if pd.notna(prod["Product_ASIN"]):
            product_image_url = f"https://images.amazon.com/images/P/{prod['Product_ASIN']}.jpg"
            st.markdown(
                f"<div style='text-align: center;'><img src='{product_image_url}' width='300' alt='{selected_product}'></div>",
                unsafe_allow_html=True
            )

        # Sentiment analysis
        product_reviews = reviews_df[reviews_df["Product_Name"] == selected_product]
        if not product_reviews.empty:
            st.header("🗣 Customer Sentiment Analysis")

            sentiment_counts = product_reviews["Sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]

            fig = px.bar(
                sentiment_counts,
                x="Sentiment",
                y="Count",
                color="Sentiment",
                color_discrete_map={"Positive": "#69f542", "Neutral": "#42ddf5", "Negative": "#f54248"},
                title=f"Sentiment Distribution for {selected_product}"
            )
            st.plotly_chart(fig, use_container_width=True)

            sentiment_map = {"Negative": 0, "Neutral": 0.5, "Positive": 1}
            product_reviews.loc[:, "Sentiment_Score_Intuitive"] = product_reviews["Sentiment"].map(sentiment_map)
            avg_score_percentage = product_reviews["Sentiment_Score_Intuitive"].mean() * 100
            st.metric(label="Customer Sentiment Score", value=f"{avg_score_percentage:.1f}%")

            st.header("⭐ Top Reviews")
            top_reviews = pd.concat([
                product_reviews[product_reviews["Sentiment"] == "Positive"].sort_values(by="Sentiment_Score", ascending=False).head(5),
                product_reviews[product_reviews["Sentiment"] == "Negative"].sort_values(by="Sentiment_Score").head(5)
            ])
            display_cols = ["Sentiment", "Review_Title", "Review_Text", "Rating", "Review_Date"]
            st.dataframe(top_reviews[display_cols].reset_index(drop=True), use_container_width=True)
        else:
            st.info("No reviews available for this product.")
    else:
        st.warning(f"Product '{selected_product}' not found in the products data.")


# ---------------- Competitor Comparison ----------------
elif section == "Competitor Comparison":
    st.markdown('<div class="main-header">Competitor Comparison</div>', unsafe_allow_html=True)
    
    st.subheader("Price vs Sentiment Analysis")
    sentiment_map = {"Negative": 0, "Neutral": 0.5, "Positive": 1}
    reviews_df.loc[:, "Sentiment_Score_Intuitive"] = reviews_df["Sentiment"].map(sentiment_map)
    avg_sentiment = reviews_df.groupby("Product_Name")["Sentiment_Score_Intuitive"].mean().reset_index()
    
    merged_df = products_df.merge(avg_sentiment, on="Product_Name", how="left").dropna(subset=["Sentiment_Score_Intuitive"])
    
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=merged_df, x="Price", y="Sentiment_Score_Intuitive", s=100, color="blue", alpha=0.7)
    X = merged_df["Price"].values.reshape(-1, 1)
    y = merged_df["Sentiment_Score_Intuitive"].values
    
    if len(X) > 1:
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        plt.plot(merged_df["Price"], y_pred, color="red", linewidth=2, label="Trend Line")
        
    plt.xlabel("Price (INR)")
    plt.ylabel("Sentiment Score")
    plt.title("Price vs Sentiment Trend Across Products")
    plt.legend()
    st.pyplot(plt)
    
    corr = merged_df["Price"].corr(merged_df["Sentiment_Score_Intuitive"])
    st.write(f"Pearson correlation between Price and Sentiment: {corr:.2f}")
    
    st.subheader("Excellent Products (Low Price + High Sentiment)")
    low_price = merged_df["Price"].quantile(0.3)
    high_sentiment = merged_df["Sentiment_Score_Intuitive"].quantile(0.7)
    excellent = merged_df[(merged_df["Price"] <= low_price) & (merged_df["Sentiment_Score_Intuitive"] >= high_sentiment)]
    st.dataframe(excellent[["Product_Name","Price","Sentiment_Score_Intuitive"]].reset_index(drop=True))
    
    st.subheader("Competitor Price Comparison")
    fig = px.bar(products_df.dropna(subset=['Price']), x="Product_Name", y="Price", color="Price", title="Competitor Price Comparison")
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(products_df[["Product_Name", "Price", "MRP", "Discount", "Rating"]])


# ---------------- Strategic Recommendations ----------------
elif section == "Strategic Recommendations":
    st.markdown('<div class="main-header">Strategic Recommendations</div>', unsafe_allow_html=True)
    
    # ... (Strategic Recommendations content remains the same) ...
    product_reviews = reviews_df[reviews_df["Product_Name"] == selected_product]
    avg_score = 0
    if not product_reviews.empty:
        sentiment_map = {"Negative": 0, "Neutral": 0.5, "Positive": 1}
        product_reviews.loc[:, "Sentiment_Score_Intuitive"] = product_reviews["Sentiment"].map(sentiment_map)
        avg_score = product_reviews["Sentiment_Score_Intuitive"].mean()

    if avg_score < 0.4:
        st.markdown("### ⚠ Needs Improvement")
        st.write("- Address negative reviews quickly\n- Offer discounts or bundles to attract buyers\n- Improve product quality if recurring complaints appear")
    elif avg_score < 0.7:
        st.markdown("### 🙂 Good Standing")
        st.write("- Maintain competitive pricing\n- Run limited-time promotions\n- Encourage happy customers to leave reviews")
    else:
        st.markdown("### 🌟 Excellent Performance")
        st.write("- Keep product quality consistent\n- Use positive reviews in marketing campaigns\n- Explore premium pricing strategies")


# ---------------- RECENT NOTIFICATIONS (New Main Section) ----------------
elif section == "Recent Notifications":
    st.markdown('<div class="main-header">🔔 Recent Notifications</div>', unsafe_allow_html=True)
    st.info("This section displays the latest alerts generated by the `notification.py` script.")

    # --- Robust Notification Log Reading (Copied from the sidebar) ---
    if os.path.exists(NOTIF_LOG):
        
        # Define columns to handle up to 18 fields to prevent ParserError
        NOTIF_COLS = [
            'Notification_ID', 'Product_ASIN', 'Product_Name', 'Notification_Type',
            'Old_Price_P1', 'Old_Price_P2', 'New_Price_P1', 'New_Price_P2',
            'Old_Stock_Status', 'New_Stock_Status',
            'Message_P1', 'Message_P2', 'Message_P3', 'Message_P4', 'Message_P5',
            'timestamp', 'Extra1', 'Extra2'
        ]
        
        try:
            notif_df = pd.read_csv(NOTIF_LOG, header=None, names=NOTIF_COLS, skiprows=1)   
            
            message_cols = [c for c in notif_df.columns if c.startswith('Message_P')]
            notif_df['message'] = notif_df[message_cols].fillna('').astype(str).agg(','.join, axis=1).str.strip(',')
            
            notif_df = notif_df.rename(columns={"Notification_Type": "type", "Product_Name": "product_name"})
            
            # --- FILTERING LOGIC ---
            valid_types = ["Price Drop", "Discount Update", "Stock Alert", "Negative Review"]
            notif_df = notif_df[notif_df['type'].isin(valid_types)]
            notif_df = notif_df[~notif_df['message'].str.startswith('http', na=False)]
            # --- END FILTERING LOGIC ---

            # --- Display All Notifications (or up to 20 for a cleaner view) ---
            notif_df = notif_df.sort_values(by="timestamp", ascending=False).head(20) # Show more than 5 in the main body

            for _, row in notif_df.iterrows():
                notif_type = row["type"]
                msg = row["message"]
                ts = row["timestamp"]
                
                product_name = str(row["product_name"]).strip()
                
                if product_name and product_name.lower() not in ('nan', ''):
                     display_title = f"{product_name} - {notif_type}" 
                else:
                     display_title = notif_type

                if "Price Drop" in notif_type:
                    color = "#f54242"
                    icon = "💸"
                elif "Negative Review" in notif_type:
                    color = "#ff9800"
                    icon = "👎"
                else:
                    color = "#2196f3"
                    icon = "ℹ️"

                # Use st.container() or st.markdown for better visualization in the main panel
                st.markdown(
                    f"""
                    <div style='background-color:{color}20;padding:15px;border-radius:8px;margin-bottom:15px;border-left: 5px solid {color};'>
                        <h4 style='margin:0;color:#333333;'><b>{icon} {display_title}</b></h4>
                        <p style='margin-top:5px;margin-bottom:5px;font-size:0.9em;color:#666666;'>
                            <small>Time: {ts}</small>
                        </p>
                        <p style='margin:0;font-size:1em;'>{msg}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        except Exception as e:
            st.error("Error loading notification log.")
            st.code(f"Details: {e}")
            
    else:
        st.info("The notification log file (`notifications.csv`) was not found. Please run `notification.py` to generate alerts.")
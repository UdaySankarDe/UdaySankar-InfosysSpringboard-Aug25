import os
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from datetime import datetime

# --- SET BASE DIRECTORY FOR ROBUST PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

# -------------------------------
# LOAD ENV
# -------------------------------
# Assuming 'env' folder is in the same directory as notification.py
load_dotenv(dotenv_path=os.path.join(BASE_DIR, "env", ".env"))

# -------------------------------
# CONFIG
# -------------------------------
# --- FIXED PATHS ---
# Assuming 'My_docs' is a folder next to notification.py
CSV_TODAY = os.path.join(BASE_DIR, "My_docs", "products.csv")
CSV_YESTERDAY = os.path.join(BASE_DIR, "My_docs", "Products_ystd.csv")

CSV_TODAY_REVIEW = os.path.join(BASE_DIR, "My_docs", "reviews_with_sentiment.csv")
CSV_YESTERDAY_REVIEW = os.path.join(BASE_DIR, "My_docs", "Reviews.csv")

NOTIF_LOG = os.path.join(BASE_DIR, "My_docs", "notifications.csv")  # 📌 dashboard will read this

PRICE_DROP_THRESHOLD = 10  # % drop
NEGATIVE_REVIEW_THRESHOLD = 2  # alerts if new negatives > this

EMAIL_SENDER = os.getenv("EMAIL_ADDRESS")
EMAIL_RECEIVER = "udaysankarde2020@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")


# -------------------------------
# EMAIL HELPER
# -------------------------------
def send_email(subject, body):
    try:
        if not all([EMAIL_SENDER, EMAIL_RECEIVER, EMAIL_PASSWORD]):
            print("⚠ Email configuration missing. Skipping email send.")
            return

        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())

        print(f"📧 Email sent → {subject}")
    except Exception as e:
        print(f"⚠ Email send failed: {e}")


# -------------------------------
# LOG TO CSV (for dashboard)
# -------------------------------
def log_notification(notif_type, message):
    os.makedirs(os.path.dirname(NOTIF_LOG), exist_ok=True)

    # Use the 'csv' module for reliable quoting, especially for the 'message' field
    # (Although pandas to_csv is generally better here if the message is already a single string)
    
    new_entry = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": notif_type,
        "message": message
    }])

    if os.path.exists(NOTIF_LOG):
        # Use mode='a' for append, header=False for no header repetition
        new_entry.to_csv(NOTIF_LOG, mode="a", header=False, index=False)
    else:
        # Create a new file with header
        new_entry.to_csv(NOTIF_LOG, index=False)

    print(f"📝 Logged notification: {notif_type}")


# -------------------------------
# PRICE DATA CLEANING HELPER
# -------------------------------
def clean_price(price_series):
    """Cleans up price strings (removes ₹, commas) and converts to numeric."""
    # Ensure it's treated as a string, then clean and replace non-numeric values with NaN
    return price_series.astype(str).str.replace('₹', '').str.replace(',', '').str.strip().replace('', pd.NA).astype(float)


# -------------------------------
# PRICE DROP CHECK
# -------------------------------
def check_price_drops():
    if not (os.path.exists(CSV_TODAY) and os.path.exists(CSV_YESTERDAY)):
        print("⚠ Missing product CSV files, skipping price check.")
        return

    try:
        today = pd.read_csv(CSV_TODAY)
        yesterday = pd.read_csv(CSV_YESTERDAY)
    except Exception as e:
        print(f"⚠ Error reading product files: {e}")
        return

    # --- Use Actual Column Names and Clean Prices ---
    # Merge on the unique product identifier (Product_ASIN)
    merged = pd.merge(today, yesterday, on="Product_ASIN", suffixes=("_today", "_yesterday"))
    
    # Apply price cleaning
    merged['Old_Price_Clean'] = clean_price(merged["Price_yesterday"])
    merged['New_Price_Clean'] = clean_price(merged["Price_today"])
    
    # Filter rows where prices are valid
    price_check_df = merged.dropna(subset=['Old_Price_Clean', 'New_Price_Clean'])

    for _, row in price_check_df.iterrows():
        try:
            old_price = row['Old_Price_Clean']
            new_price = row['New_Price_Clean']

            if old_price > 0:
                drop_percent = ((old_price - new_price) / old_price) * 100
                if drop_percent >= PRICE_DROP_THRESHOLD:
                    body = (
                        f"Price Drop Alert 🚨\n\n"
                        f"Product: {row['Product_Name_today']}\n"
                        f"Old Price: ₹{old_price:,.2f}\n"  # Format for display
                        f"New Price: ₹{new_price:,.2f}\n"  # Format for display
                        f"Drop: {drop_percent:.2f}%\n\n"
                        f"Check competitor site immediately."
                    )
                    subject = f"⚠ Price Drop: {row['Product_Name_today']}"
                    send_email(subject, body)
                    log_notification("Price Drop", body)
        except Exception as e:
            print(f"⚠ Error checking price for {row.get('Product_Name_today')}: {e}")


# -------------------------------
# NEGATIVE REVIEW CHECK
# -------------------------------
def check_negative_reviews():
    if not (os.path.exists(CSV_TODAY_REVIEW) and os.path.exists(CSV_YESTERDAY_REVIEW)):
        print("⚠ Missing review CSV files, skipping review check.")
        return
        
    try:
        today = pd.read_csv(CSV_TODAY_REVIEW)
        yesterday = pd.read_csv(CSV_YESTERDAY_REVIEW)
    except Exception as e:
        print(f"⚠ Error reading review files: {e}")
        return

    # --- Use Actual Column Names ---
    # Concatenate and drop duplicates based on Product_ASIN and Review_Text to find NEW reviews.
    merged = pd.concat([yesterday, today]).drop_duplicates(
        subset=["Product_ASIN", "Review_Text"], # Assuming this combo is unique for a review
        keep=False # Keeps only rows that are NOT duplicated (i.e., new reviews)
    )

    # Convert rating to numeric, filter for ratings 1 and 2
    merged['Rating'] = pd.to_numeric(merged['Rating'], errors='coerce')
    negatives = merged[merged["Rating"].isin([1.0, 2.0])]

    if len(negatives) >= NEGATIVE_REVIEW_THRESHOLD:
        body = (
            f"Negative Review Alert 🚨\n\n"
            f"Found {len(negatives)} new negative reviews today.\n\n"
            f"Example:\n"
            f"Product: {negatives.iloc[0]['Product_Name']}\n"
            f"Review: {negatives.iloc[0]['Review_Text']}\n"
            f"Rating: {negatives.iloc[0]['Rating']:.1f}\n\n"
            f"Check full review report for details."
        )
        send_email("⚠ Negative Reviews Detected", body)
        log_notification("Negative Review", body)


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    print("🔍 Running notification checks...")
    check_price_drops()
    check_negative_reviews()
    print("✅ Notification run complete.")
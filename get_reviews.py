import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import csv

# Load products.csv
products_df = pd.read_csv(r"C:\Users\Uday Sankar De\Desktop\Springboard intern 2025\Amazon\products.csv")

all_reviews = []

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

def scrape_reviews(asin, reviews_url, max_pages=5):
    reviews = []
    for page in range(1, max_pages+1):
        url = f"{reviews_url}?pageNumber={page}"
        print(f"🔎 Scraping {asin} - page {page} ...")
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"❌ Failed to fetch page {page} for {asin}")
            break

        soup = BeautifulSoup(response.text, "html.parser")

        review_blocks = soup.select("div[data-hook='review']")
        if not review_blocks:
            print(f"⚠️ No reviews found on page {page} for {asin}")
            break

        for rb in review_blocks:
            title = rb.select_one("a[data-hook='review-title']")
            title = title.text.strip() if title else ""

            text = rb.select_one("span[data-hook='review-body']")
            text = text.text.strip() if text else ""

            rating = rb.select_one("i[data-hook='review-star-rating']")
            rating = rating.text.strip() if rating else ""

            reviewer = rb.select_one("span.a-profile-name")
            reviewer = reviewer.text.strip() if reviewer else ""

            date = rb.select_one("span[data-hook='review-date']")
            date = date.text.strip() if date else ""

            helpful = rb.select_one("span[data-hook='helpful-vote-statement']")
            helpful = helpful.text.strip() if helpful else "0"

            reviews.append({
                "Product_ASIN": asin,
                "Review_Title": title,
                "Review_Text": text,
                "Rating": rating,
                "Reviewer": reviewer,
                "Review_Date": date,
                "Helpful_Count": helpful,
                "Review_Link": url
            })

        print(f"✅ Collected {len(reviews)} reviews so far for {asin}")
        time.sleep(random.uniform(1, 3))  # avoid blocking

    return reviews


# Loop over each product
for _, row in products_df.iterrows():
    asin = row["Product_ASIN"]
    reviews_url = row["Reviews_Link"]
    print(f"\n=== Processing {asin} ===")
    product_reviews = scrape_reviews(asin, reviews_url, max_pages=3)
    all_reviews.extend(product_reviews)
    print(f"📦 Finished {asin}: {len(product_reviews)} reviews collected")

# Save to reviews.csv
reviews_df = pd.DataFrame(all_reviews)
reviews_df.to_csv("reviews.csv", index=False, quoting=csv.QUOTE_ALL)

print(f"\n🎉 Saved {len(all_reviews)} total reviews to reviews.csv")

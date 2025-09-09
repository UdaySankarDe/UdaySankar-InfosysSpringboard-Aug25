import os
import time
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
HEADERS = json.loads(os.getenv("HEADERS"))
COOKIES = json.loads(os.getenv("COOKIES"))
BASE_URL = "https://www.amazon.in/s?k=laptops"

def fetch_content(url):
    
    try:
        response = requests.get(url, headers=HEADERS, cookies=COOKIES, timeout=10)
        response.raise_for_status()
        print(f"Successfully fetched page: {url}")
        soup = BeautifulSoup(response.content, "html.parser")
        return soup
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page content: {e}")
        return None

def get_title(item):
    title = item.find("h2", class_="a-size-medium a-spacing-none a-color-base a-text-normal")
    return title.text.strip() if title else None

def get_brand(title_text):
    if title_text.startswith("soundcore"):
        return "Anker"
    return title_text.split()[0] if title_text else None

def get_price(item):
    discount_price = item.find("span", class_="a-price")
    return (
        discount_price.find("span", class_="a-offscreen").text.strip()
        if discount_price and discount_price.find("span", class_="a-offscreen")
        else None
    )

def get_mrp(item):
    base_price = item.find("div", class_="a-section aok-inline-block")
    return (
        base_price.find("span", class_="a-offscreen").text.strip()
        if base_price and base_price.find("span", class_="a-offscreen")
        else None
    )

def get_discount_percentage(item):
    discount = item.find("span", string=lambda text: text and "%" in text)
    return discount.text.strip().strip("()") if discount else None

def get_rating(item):
    rating = item.find("span", class_="a-icon-alt")
    return rating.text.strip() if rating else None

def get_reviews(item):
    reviews = item.find("span", class_="a-size-base s-underline-text")
    return reviews.text.strip() if reviews else None

def get_product_id(item):
    return item.get("data-asin", None)

def get_product_link(item):
    
    link = item.find("a", class_="a-link-normal s-no-outline")
    return "https://www.amazon.in" + link["href"] if link and "href" in link.attrs else None

def get_delivery_date(item):
    delivery = item.find("span", class_="a-text-bold")
    if not delivery:
        delivery = item.find("span", string=lambda text: text and "delivery" in text.lower())
    return delivery.text.strip() if delivery else None

def get_service(item):
    service = item.find("span", class_="a-text-bold", string=lambda text: "Service" in text if text else False)
    if not service:
        service = item.find("span", string=lambda text: text and "Service:" in text)
    return service.text.strip() if service else None



def parse_product(item):
    
    try:
        title_text = get_title(item)
        product_link = get_product_link(item)
        return {
            "Product_Name": title_text,
            "Product_ASIN": get_product_id(item),
            "Brand": get_brand(title_text),
            "Price": get_price(item),
            "MRP": get_mrp(item),
            "Discount": get_discount_percentage(item),
            "Rating": get_rating(item),
            "Reviews": get_reviews(item),
            "Product_Link": product_link,
            "Delivery_Date": get_delivery_date(item),
            "Service": get_service(item),
            "Scraped_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        print(f"Error parsing product: {e}")
        return None

def scrape_page(url):
    
    soup = fetch_content(url)
    if not soup:
        return [], None

    items = soup.find_all("div", {"data-component-type": "s-search-result"})
    products = []

    for item in items:
        product_data = parse_product(item)
        if product_data:
            products.append(product_data)

    next_button = soup.find("a", class_="s-pagination-next")
    next_page_url = "https://www.amazon.in" + next_button["href"] if next_button and "href" in next_button.attrs else None

    return products, next_page_url

def scrape_within_time(base_url, max_time_minutes=5):
    
    all_products = []
    current_page = 1
    next_page_url = base_url
    end_time = datetime.now() + timedelta(minutes=max_time_minutes)

    try:
        print(f"Starting the scraping process for {max_time_minutes} minutes...")
        while next_page_url and datetime.now() < end_time:
            print(f"Scraping page {current_page}...")
            products, next_page_url = scrape_page(next_page_url)
            
            if not products:
                print("No more products found. Stopping scraping.")
                break

            all_products.extend(products)
            print(f"Scraped {len(products)} products from page {current_page}.")
            current_page += 1
            time.sleep(2)  # Add delay to prevent being blocked by the server
            
            if datetime.now() >= end_time:
                print("Time limit reached. Stopping scraping.")
                break

    except Exception as e:
        print(f"An error occurred during scraping: {e}")
    
    print("Scraping finished.")
    return all_products

def save_to_csv(data, filename="product.csv"):
    
    try:
        directory = "Amazon"
        
        full_file_path = os.path.join(directory, filename)
        
        df = pd.DataFrame(data)
        df.to_csv(full_file_path, index=False)
        
        print(f"Data successfully saved to {full_file_path}.")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")


if __name__ == "__main__":
    max_scraping_time = 1  # Max time in minutes
    print("Starting the scraping process...")
    products = scrape_within_time(BASE_URL, max_time_minutes=max_scraping_time)

    if products:
        save_to_csv(products)
        print(f"Scraped {len(products)} products.")
    else:
        print("No products found.")
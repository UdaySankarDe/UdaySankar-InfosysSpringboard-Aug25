# 🛒 E-Commerce Competitor Strategy Dashboard
*A data-driven tool for analyzing competitors, pricing, and sentiment in e-commerce.*

This Python-powered dashboard empowers businesses to **scrape product data, analyze customer sentiment, compare competitors, and query insights using Retrieval-Augmented Generation (RAG)** — all in one place.

---

## 🚀 Features
- 🔍 **Automated Data Collection** — Scrape product listings and reviews using **Selenium** and **BeautifulSoup**.  
- 📊 **Data Processing & Sentiment Analysis** — Clean and analyze data with **pandas** + **TextBlob**.  
- 📈 **Interactive Dashboard** — Visualize pricing, discounts, and sentiment via **Streamlit**.  
- ⚔️ **Competitor Comparison** — Compare brands on metrics like price, rating, and availability.  
- 💡 **Strategic Recommendations** — Get auto-generated insights for market positioning.  
- 📬 **Email Alerts** — Receive notifications for major competitor updates.  
- 🧠 **RAG-Powered Q&A** — Query internal documents using **LangChain**, **FAISS**, and **Google GenAI/Groq**.  

---

## 🛠️ Tech Stack
| Category | Technologies |
|-----------|--------------|
| **Language** | Python |
| **Web Scraping** | Selenium, BeautifulSoup |
| **Data Processing** | pandas, numpy |
| **Sentiment Analysis** | TextBlob |
| **Dashboard** | Streamlit |
| **Visualization** | Plotly, Matplotlib |
| **Document Q&A (RAG)** | FAISS, LangChain, Google GenAI, Groq |
| **Alerts** | SMTP Email |

---

## 📂 Project Structure
```
infosys_internship/
├── env/
├── faiss_index/
├── My_docs/
│   ├── notifications.csv
│   ├── Products_ystd.csv
│   ├── products.csv
│   ├── reviews_with_sentiment.csv
│   └── Reviews.csv
├── .env
├── .gitignore
├── analyse_sentiment.py       # Performs sentiment analysis
├── app.py                     # Streamlit dashboard entry point
├── competitor.py              # Competitor comparison logic
├── config.py                  # Environment and API configurations
├── get_reviews.py             # Scrapes and stores product reviews
├── LICENSE                    
├── main.py                    # Ask questions to RAG
├── notification.py            # Sends email alerts
├── products.py                # Product scraping logic
├── rag.md                     # Documentation for RAG usage
├── rag.py                     # RAG pipeline (FAISS + LangChain + GenAI)
├── README.md                  # Project overview and guide
├── requirements.txt           # Dependency list
├── setup.md                   # Setup and environment instructions
├── web_scraping.py            # Combined scraping + analysis script
└── sheets.md                  # Google Sheets integration notes
```

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/UdaySankarDe/UdaySankar-InfosysSpringboard-Aug25.git
cd UdaySankar-InfosysSpringboard-Aug25
```

### 2️⃣ (Optional) Create a virtual environment
```bash
python -m venv myenv
```
Activate the environment:

**macOS/Linux**
```bash
source myenv/bin/activate
```

**Windows**
```bash
myenv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure environment variables
Create a `.env` file in the root directory and add:
```bash
EMAIL_USER="your_email@example.com"  #amazon app
EMAIL_PASS="your_app_password"       #amazon password
GOOGLE_API_KEY="your_google_genai_key"
GROQ_API_KEY="your_grow_api_key"
```

---

## ▶️ Usage

### 🧩 Run the main dashboard
```bash
streamlit run app.py
```

### 📄 Document Q&A (RAG)
Place your `.txt`, `.pdf`, `.csv`, or `.md` files inside the **My_docs/** folder.

Then run:
```bash
python rag.py --query "Summarize competitor pricing strategies"
```

### 📬 Email Alerts
Email notifications are automatically triggered for key competitor events — such as sudden price changes, rating shifts, or new product listings.

---

## 🧠 Example Use Cases

| Use Case | Description |
|-----------|-------------|
| **Product Analysis** | Explore and compare pricing, discounts, ratings, and stock availability. |
| **Customer Sentiment** | Analyze sentiment trends from real customer reviews. |
| **Competitor Comparison** | Visualize multi-brand insights across key metrics. |
| **Document Q&A** | Ask questions like “Summarize the reviews” or “What are competitors’ pricing strategies?”. |
| **Strategic Recommendations** | Receive data-driven insights for smarter market positioning. |

---

## 🧠 How RAG Works
This system integrates **FAISS vector search**, **LangChain orchestration**, and **Google GenAI/Groq models** for intelligent document understanding.  
It retrieves relevant passages, then uses an LLM to answer complex business questions contextually and accurately.

---

## 📄 License
This project is licensed under the **MIT License** — see the LICENSE file for details.

---

## 🔑 Keywords
e-commerce competitor analysis, sentiment analysis, dashboard, Streamlit, web scraping, RAG, FAISS, LangChain, AI analytics

---


✨ **Developed by**
**Uday Sankar De**  
📧 udaysankarde2022@gmail.com

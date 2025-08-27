import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain core + utilities
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loaders
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader

# Vector store
from langchain_community.vectorstores import FAISS

# Embeddings + LLMs
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# RAG chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# ───────────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────
DOCS_PATH     = Path(r"C:\Users\Uday Sankar De\Desktop\Infy Internship 2025\UdaySankar-InfosysSpringboard-Aug25\data")
         # folder or file
INDEX_PATH    = Path("./faiss_index")      # FAISS storage
REBUILD_INDEX = True                       # toggle this
EMBED_MODEL   = "models/embedding-001"     # Gemini embeddings
CHAT_MODEL    = "gemini-1.5-flash"         # Gemini chat model
TOP_K         = 4
SEARCH_TYPE   = "mmr"                      # mmr | similarity
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 120
QUESTION      = "Give me a 2-line summary of the docs and cite sources."


# ───────────────────────────────────────────────────────────
# STEP 1) Ensure API Keys
# ───────────────────────────────────────────────────────────
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise SystemExit("❌ GOOGLE_API_KEY not set in .env")
if not os.getenv("GROQ_API_KEY"):
    print("⚠️ GROQ_API_KEY not found (you can still run with Gemini only)")


# # ───────────────────────────────────────────────────────────
# STEP 2) Load documents
# ───────────────────────────────────────────────────────────
def find_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    exts = {".txt", ".md", ".pdf", ".csv"}
    return [p for p in path.rglob("*") if p.suffix.lower() in exts]

def load_documents(paths: list[Path]) -> list[Document]:
    docs: list[Document] = []
    for p in paths:
        try:
            if p.suffix.lower() in {".txt", ".md"}:
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
            elif p.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(p)).load())
            elif p.suffix.lower() == ".csv":
                import pandas as pd

                df = pd.read_csv(str(p))

                # ⚡ Limit rows for testing (change 10 as needed)
                df = df.head(20)

                # Use Title + Desc
                if "Title" in df.columns and "Desc" in df.columns:
                    for _, row in df.iterrows():
                        title = str(row["Title"])
                        desc = str(row["Desc"])
                        content = f"Product: {title}\nDescription: {desc}"
                        metadata = {"source": str(p), "title": title}
                        docs.append(Document(page_content=content, metadata=metadata))
                else:
                    print(f"[WARN] CSV {p} missing Title or Desc column.")
        except Exception as e:
            print(f"[WARN] Could not load {p}: {e}")
    return docs





# ───────────────────────────────────────────────────────────
# STEP 3) Split documents
# ───────────────────────────────────────────────────────────
def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


# ───────────────────────────────────────────────────────────
# STEP 4) Build / load FAISS index
# ───────────────────────────────────────────────────────────
def build_or_load_faiss(chunks: list[Document], rebuild: bool) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

    if rebuild:
        print("🔁 Building FAISS index...")
        vs = FAISS.from_documents(chunks, embeddings)
        INDEX_PATH.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(INDEX_PATH))
        return vs

    print("📦 Loading FAISS index...")
    return FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )


# ───────────────────────────────────────────────────────────
# STEP 5) Make retriever
# ───────────────────────────────────────────────────────────
def make_retriever(vectorstore: FAISS):
    return vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={"k": TOP_K}
    )


# ───────────────────────────────────────────────────────────
# STEP 6) Build RAG chain
# ───────────────────────────────────────────────────────────
def make_rag_chain(retriever):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise assistant. Answer ONLY from context. "
                   "If not in context, say 'I don't know'. Cite sources."),
        ("human", "Question:\n{input}\n\nContext:\n{context}")
    ])

    # Choose LLM: Gemini or Groq
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)
    # Or uncomment this to use Groq instead:
    # llm = ChatGroq(model="mixtral-8x7b-32768")

    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, doc_chain)


# ───────────────────────────────────────────────────────────
# STEP 7) Pretty-print sources
# ───────────────────────────────────────────────────────────
def format_sources(ctx: list[Document]) -> str:
    lines = []
    for d in ctx:
        src = d.metadata.get("source") or "unknown"
        page = d.metadata.get("page")
        name = Path(src).name
        lines.append(f"- {name}" + (f" (page {page})" if page is not None else ""))
    return "\n".join(lines)


# ───────────────────────────────────────────────────────────
# STEP 8) Main
# ───────────────────────────────────────────────────────────
def main():
    chunks = []
    if REBUILD_INDEX:
        files = find_files(DOCS_PATH)
        if not files:
            raise SystemExit("❌ No valid files found in docs path")
        docs = load_documents(files)
        chunks = split_documents(docs)

    vectorstore = build_or_load_faiss(chunks, rebuild=REBUILD_INDEX)
    retriever = make_retriever(vectorstore)
    rag = make_rag_chain(retriever)

    if QUESTION:
        result = rag.invoke({"input": QUESTION})
        print("\n🧠 Answer:\n", result.get("answer"))
        if "context" in result:
            print("\n📚 Sources:\n", format_sources(result["context"]))


if __name__ == "__main__":
    main()

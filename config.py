from pathlib import Path


DOCS_PATH     = Path(r"C:\Users\Uday Sankar De\Desktop\Infy Internship 2025\UdaySankar-InfosysSpringboard-Aug25\data")
         # folder or file
INDEX_PATH    = Path("./faiss_index")      # FAISS storage
REBUILD_INDEX = True                       # toggle this
EMBED_MODEL   = "models/embedding-001"     # Gemini embeddings
CHAT_MODEL    = "gemini-1.5-flash"         # Gemini chat model

TOP_K         = 20                         # how many chunks to retrieve
SEARCH_TYPE   = "mmr"                      # "mmr" | "similarity" | "similarity_score_threshold"
CHUNK_SIZE    = 300
CHUNK_OVERLAP = 20

QUESTION      = "List all products from the Sports category with their names, descriptions, and prices."
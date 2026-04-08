import os
import json
import urllib.request
import urllib.error
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from pypdf import PdfReader
from io import BytesIO

from inference import run_inference, API_BASE_URL, MODEL_NAME, HF_TOKEN

HF_API_KEY = HF_TOKEN
HF_EMBED_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-large-en-v1.5"

class HuggingFaceCustomEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        }
        # HF Inference API expects a list of strings
        payload = {
            "inputs": input,
            "options": {"wait_for_model": True}
        }
        req = urllib.request.Request(
            HF_EMBED_URL, 
            data=json.dumps(payload).encode("utf-8"), 
            headers=headers
        )
        try:
            with urllib.request.urlopen(req) as f:
                res = json.loads(f.read().decode("utf-8"))
                if not isinstance(res, list):
                    print("Unexpected response:", res)
                    raise ValueError(str(res))
                
                # Check dimensionality:
                # If 1D: Return [res]
                # If 2D (batch_size, dim): Return res
                # If 3D (batch_size, seq_len, dim): Return CLS token config [doc[0] for doc in res]
                
                # 1D case (single input flattened)
                if len(res) > 0 and isinstance(res[0], float):
                    res = [res]
                    
                final_res = []
                for item in res:
                    if isinstance(item, list) and len(item) > 0 and isinstance(item[0], list):
                        # 3D: item is sequence of vectors. We take the CLS token (first token)
                        final_res.append(item[0])
                    else:
                        # 2D: item is a vector
                        final_res.append(item)
                
                # STRICT DIMENSION CHECK TO PREVENT CHROMADB PANIC
                for emb in final_res:
                    if not isinstance(emb, list) or len(emb) != 1024:
                        print("CRASH AVERTED: Invalid returned embedding shape:", len(emb) if isinstance(emb, list) else type(emb))
                        raise ValueError("HF API returned invalid dimension length instead of 1024")
                return final_res
        except Exception as e:
            print(f"Error calling HuggingFace Embed API: {e}")
            raise

# Use /data/ for persistent storage on HF Spaces, fallback for local dev
DATA_DIR = "/data" if os.path.isdir("/data") else "."
os.makedirs(DATA_DIR, exist_ok=True)
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
custom_ef = HuggingFaceCustomEmbeddingFunction()
# Important: New path and collection name so it rebuilds vectors from scratch correctly using the new model dims
collection = chroma_client.get_or_create_collection(
    name="docs",
    embedding_function=custom_ef
)

def get_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\\n"
    return text

def get_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode('utf-8', errors='ignore')

def chunk_text(text: str, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def process_and_store_document(user_id: int, doc_id: int, file_bytes: bytes, filename: str):
    if filename.lower().endswith('.pdf'):
        text = get_text_from_pdf(file_bytes)
    elif filename.lower().endswith('.txt'):
        text = get_text_from_txt(file_bytes)
    else:
        raise ValueError("Unsupported file format")
    
    chunks = chunk_text(text)
    if not chunks:
        return
        
    ids = [f"doc_{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"user_id": user_id, "doc_id": doc_id, "filename": filename} for _ in chunks]
    
    # Store in chroma
    embs = custom_ef(chunks)
    with open("debug_log.txt", "w") as f:
        f.write(f"DEBUG ADD: docs={len(chunks)}, ids={len(ids)}\n")
        f.write(f"DEBUG ADD EMBS: len={len(embs)}, inner={len(embs[0]) if embs and isinstance(embs[0], list) else '?'}\n")
    try:
        collection.add(
            embeddings=embs,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        print("DEBUG ADD SUCCESS")
    except Exception as e:
        print("DEBUG ADD EXCEPTION:", e)
        raise e

def retrieve_context(user_id: int, query: str, top_k: int = 4):
    print(f"DEBUG QUERY: {query}")
    embs = custom_ef([query])
    with open("debug_log.txt", "a") as f:
        f.write(f"DEBUG QUERY: {query}\n")
        f.write(f"DEBUG QUERY EMBS: len={len(embs)}, inner={len(embs[0]) if embs and isinstance(embs[0], list) else '?'}\n")
    try:
        results = collection.query(
            query_embeddings=embs,
            n_results=top_k,
            where={"user_id": user_id}
        )
        print("DEBUG QUERY SUCCESS")
    except Exception as e:
        print("DEBUG QUERY EXCEPTION:", e)
        raise e
    if not results['documents'] or not results['documents'][0]:
        return []
    
    context_chunks = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        context_chunks.append(f"Source: {meta['filename']}\\nContent: {doc}")
    return context_chunks

def answer_query(user_id: int, query: str) -> str:
    context_chunks = retrieve_context(user_id, query)
    # Call the compliant inference function from our root inference.py
    # This ensures we use the OpenAI client and print the required [START]/[STEP]/[END] logs
    answer = run_inference(query, context_chunks)
    return answer

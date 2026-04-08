from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from database import (init_db, create_user, get_user_by_username, get_user_by_api_key, 
                      add_document, get_user_documents, verify_password, 
                      create_session, get_user_by_session, delete_session)
from rag import process_and_store_document, answer_query
from inference import openenv_reset, openenv_validate

# OpenEnv Integration
from openenv.core.env_server import create_fastapi_app
from your_environment import YourEnvironment
from models import YourAction, YourObservation

load_dotenv()

# --- OpenEnv Setup ---
# We use the factory function as required by the hackathon guidelines
env_instance = YourEnvironment()
app = create_fastapi_app(env_instance, YourAction, YourObservation)

# Modify the app created by the factory to include our RAG features
app.title = "DocKey AI"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

init_db()

# --- Dependencies ---
def get_current_user_from_cookie(request: Request):
    token = request.cookies.get("session")
    if not token:
        return None
    return get_user_by_session(token)

# --- Web Routes (Existing - Unchanged) ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, error: str = None):
    user = get_current_user_from_cookie(request)
    if user:
        return RedirectResponse(url="/dashboard")
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request, "error": error})

@app.post("/register")
async def register(request: Request, username: str = Form(...), password: str = Form(...)):
    user_id, api_key = create_user(username, password)
    if not user_id:
        return RedirectResponse(url="/?error=Username already taken", status_code=303)
    
    session_token = create_session(user_id)
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(key="session", value=session_token, httponly=True, samesite="none", secure=True)
    return response

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = get_user_by_username(username)
    if not user or not verify_password(password, user["password_hash"]):
        return RedirectResponse(url="/?error=Invalid credentials", status_code=303)
        
    session_token = create_session(user["id"])
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(key="session", value=session_token, httponly=True, samesite="none", secure=True)
    return response

@app.get("/logout")
async def logout(request: Request):
    token = request.cookies.get("session")
    if token:
        delete_session(token)
    response = RedirectResponse(url="/")
    response.delete_cookie("session")
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, error: str = None):
    user = get_current_user_from_cookie(request)
    if not user:
        return RedirectResponse(url="/")
        
    docs = get_user_documents(user["id"])
    return templates.TemplateResponse(request=request, name="dashboard.html", context={"request": request, "user": user, "documents": docs, "error": error})

@app.post("/upload")
async def upload_document(request: Request, file: UploadFile = File(...)):
    user = get_current_user_from_cookie(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    if not file.filename.lower().endswith(('.pdf', '.txt')):
        return RedirectResponse(url="/dashboard?error=Unsupported file type", status_code=303)
        
    contents = await file.read()
    doc_id = add_document(user["id"], file.filename)
    
    try:
        process_and_store_document(user["id"], doc_id, contents, file.filename)
    except Exception as e:
        print(f"Error processing document: {e}")
        return RedirectResponse(url="/dashboard?error=Failed to process document", status_code=303)
        
    return RedirectResponse(url="/dashboard", status_code=303)

# --- API Routes for Chatbot ---
class ChatRequest(BaseModel):
    query: str

def get_user_from_api_keyHeader(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="I can only access documents associated with the current account or API key.")
    
    api_key = auth_header.split(" ")[1]
    user = get_user_by_api_key(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="I can only access documents associated with the current account or API key.")
    return user

@app.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request):
    user = get_user_from_api_keyHeader(request)
    answer = answer_query(user["id"], req.query)
    return {"answer": answer}

@app.post("/web/chat")
async def web_chat(req: ChatRequest, request: Request):
    user = get_current_user_from_cookie(request)
    if not user:
        return JSONResponse(status_code=401, content={"answer": "I can only access documents associated with the current account or API key."})
        
    answer = answer_query(user["id"], req.query)
    return {"answer": answer}

# --- OpenEnv Endpoints (Mandatory for Hackathon) ---
@app.post("/openenv/reset")
@app.post("/reset_health") # Alias to distinguish from RL reset
async def endpoint_openenv_reset():
    """Endpoint for hackathon checker to reset environment"""
    try:
        result = openenv_reset()
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/openenv/validate")
@app.get("/validate")
async def endpoint_openenv_validate():
    """Endpoint for hackathon checker to validate environment"""
    try:
        result = openenv_validate()
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# --- RL Environment Endpoints (via Factory) ---
# Note: create_fastapi_app already adds /reset, /step, /state by default.
# We have integrated these into the main 'app' object.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)

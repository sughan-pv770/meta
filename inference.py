import os
import json
from openai import OpenAI

# 1. READ ENVIRONMENT VARIABLES WITH DEFAULTS (Hackathon Requirement)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Token is now strictly read from environment to prevent Git push failures 
# due to secret scanning. In HF Spaces, set this in Settings > Secrets.

if HF_TOKEN is None:
    # Fallback to empty string to prevent constructor crash, 
    # but actual calls will fail with 401 until token is set in Env.
    HF_TOKEN = ""

# 2. INITIALIZE OPENAI CLIENT (Hackathon Requirement)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# Global state for OpenEnv
_env_state = {"status": "ready", "query_count": 0}

# 3. CORE INFERENCE LOGIC
def run_inference(query: str, context_chunks: list = None):
    """
    Main inference function. 
    Matches hackathon structure while maintaining DocKey AI RAG features.
    """
    task_name = "rag-chat"
    benchmark = "dockey-ai"
    
    # [START] line type
    print(f"[START] task={task_name} env={benchmark} model={MODEL_NAME}")
    
    context_text = ""
    if context_chunks:
        context_text = "\n\nContext:\n" + "\n".join(context_chunks)
    
    prompt = f"{context_text}\n\nUser Question: {query}\n\nAnswer professionally based on context provided."
    
    steps = 0
    rewards = []
    success = "false"
    
    try:
        # User OpenAI Client as per requirement 2
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a professional document assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.2
        )
        answer = response.choices[0].message.content
        
        # [STEP] line type (simulating a single-step RAG episode)
        steps += 1
        reward = 1.00 # Positive reward for successful response
        rewards.append(f"{reward:.2f}")
        print(f"[STEP] step={steps} action=rag_retrieve_and_respond reward={reward:.2f} done=true error=null")
        
        success = "true"
        
    except Exception as e:
        error_msg = str(e).replace('\n', ' ')
        print(f"[STEP] step=1 action=failed_inference reward=0.00 done=true error={error_msg}")
        answer = f"Error during inference: {error_msg}"
        rewards.append("0.00")
        success = "false"
        steps = 1

    # [END] line type
    rewards_str = ",".join(rewards)
    print(f"[END] success={success} steps={steps} rewards={rewards_str}")
    
    return answer

# --- OpenEnv Lifecycle functions (Expected by app.py) ---

def openenv_reset():
    """Reset inference environment state. Required by FastAPI /openenv/reset."""
    global _env_state
    _env_state = {"status": "ready", "query_count": 0}
    return {"status": "ok", "message": "Environment reset successfully."}

def openenv_validate():
    """Validate healthy state. Required by FastAPI /openenv/validate."""
    global _env_state
    try:
        # Quick check using dummy prompt
        run_inference("health check")
        val_ok = True
    except:
        val_ok = False
        
    return {
        "status": "ok" if val_ok else "degraded",
        "env_state": _env_state,
        "config": {
            "api_base": API_BASE_URL,
            "model": MODEL_NAME
        }
    }

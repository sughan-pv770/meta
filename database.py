import sqlite3
import bcrypt
import secrets
import os

# Use /data/ for persistent storage on HF Spaces, fallback for local dev
DATA_DIR = "/data" if os.path.isdir("/data") else "."
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "dockey.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            api_key TEXT UNIQUE NOT NULL
        )
    ''')
    # Documents table
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            status TEXT DEFAULT 'processed',
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    # Sessions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def generate_api_key() -> str:
    return "dk_" + secrets.token_hex(16)

def create_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        api_key = generate_api_key()
        c.execute("INSERT INTO users (username, password_hash, api_key) VALUES (?, ?, ?)",
                  (username, hash_password(password), api_key))
        user_id = c.lastrowid
        conn.commit()
        return user_id, api_key
    except sqlite3.IntegrityError:
        return None, None
    finally:
        conn.close()

def get_user_by_username(username):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return dict(user) if user else None

def get_user_by_id(user_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = c.fetchone()
    conn.close()
    return dict(user) if user else None

def get_user_by_api_key(api_key):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE api_key = ?", (api_key,))
    user = c.fetchone()
    conn.close()
    return dict(user) if user else None

def add_document(user_id, filename):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO documents (user_id, filename) VALUES (?, ?)", (user_id, filename))
    doc_id = c.lastrowid
    conn.commit()
    conn.close()
    return doc_id

def get_user_documents(user_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM documents WHERE user_id = ?", (user_id,))
    docs = c.fetchall()
    conn.close()
    return [dict(doc) for doc in docs]

def create_session(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    session_token = secrets.token_urlsafe(32)
    c.execute("INSERT INTO sessions (session_token, user_id) VALUES (?, ?)", (session_token, user_id))
    conn.commit()
    conn.close()
    return session_token

def get_user_by_session(session_token):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT u.* FROM users u JOIN sessions s ON u.id = s.user_id WHERE s.session_token = ?", (session_token,))
    user = c.fetchone()
    conn.close()
    return dict(user) if user else None

def delete_session(session_token):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM sessions WHERE session_token = ?", (session_token,))
    conn.commit()
    conn.close()

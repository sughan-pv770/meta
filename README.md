---
title: DocKey AI
emoji: 🔑
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# DocKey AI

DocKey AI is a secure multi-tenant RAG chatbot built with FastAPI, ChromaDB, and Hugging Face APIs.

## Features
- User registration and login with session management
- Per-user API key for external programmatic access
- Upload PDF/TXT documents to your private knowledge base
- RAG-powered chat grounded entirely in your uploaded documents
- Strict multi-tenant isolation — one user cannot access another's data

## Using the API
```bash
curl -X POST "https://parasuramane24-devkey-ai.hf.space/api/chat" \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"query": "What does my document say about X?"}'
```

# Kavach (Gemini)

UPI safety inquiry chatbot: FastAPI + Gemini REST API + SQLite.

    pip install -r requirements.txt
    copy .env.example .env     (then add GEMINI_API_KEY)
    python -m uvicorn app:app --reload --port 8000

Open http://127.0.0.1:8000 . Tests: python run_tests.py (server running).
Edit knowledge.md to change what the bot knows.

# Backend (LangGraph)

## 1) Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -e .
copy .env.example .env
```

Add your `GOOGLE_API_KEY` to `.env`.

## 2) Run


```bash
uvicorn app.main:app --reload --reload-dir app --port 8000

python -m app.main rebuild
```
Test at `http://localhost:8000/health`.


                START
                   │
                   ▼
              +---------+
              | chatbot |
              +---------+
                   │
          tool requested?
           /           \
         no             yes
         │               │
         ▼               ▼
        END         +---------+
                    |  tools  |
                    +---------+
                         │
                         ▼
                    +---------+
                    | chatbot |
                    +---------+
                         │
          asks for Google?
           /            \
         no              yes
         │                │
         ▼                ▼
        END        search already?
                    /          \
                  yes          no
                  │             │
                  ▼             ▼
             chatbot      google_search
                               │
                               ▼
                          chatbot
                               │
                               ▼
                              END
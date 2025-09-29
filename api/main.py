from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="kamlan API", version="0.1.0")

# Allow Firebase Hosting origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kamlan.web.app",
        "https://*.web.app",
        "https://*.firebaseapp.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/api/hello")
def hello(name: str = Query("world")):
    return {"message": f"Hello, {name}!"}


@app.get("/api/analyze")
def analyze(file_url: str = Query(None)):
    # Placeholder response; implement real logic later
    return {"ok": True, "file_url": file_url, "note": "Analysis stub; backend scaffold is working."}



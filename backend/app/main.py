from fastapi import FastAPI

app = FastAPI(title="Upload API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def main():
    return {"status": "ok"}



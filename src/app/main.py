from fastapi import FastAPI

from src.app.api.routes import daily_logs, predictions
from src.app.db.database import  Base, engine

app = FastAPI()

Base.metadata.create_all(engine)

app.include_router(predictions.router)
app.include_router(daily_logs.router)

@app.get("/health")
def health():
    return {"status": "ok"}
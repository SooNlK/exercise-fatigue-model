from fastapi import Depends, FastAPI

from src.app.api.dependencies import verify_browser_request
from src.app.api.routes import daily_logs, predictions, register, login, me
from src.app.db.database import  Base, engine

app = FastAPI(
    dependencies=[Depends(verify_browser_request)],
)

Base.metadata.create_all(engine)

app.include_router(register.router)
app.include_router(login.router)
app.include_router(me.router)
app.include_router(predictions.router)
app.include_router(daily_logs.router)

@app.get("/health")
def health():
    return {"status": "ok"}

import uvicorn
from fastapi import FastAPI
from app.api.routes import router
from app.resources.database import Database

app = FastAPI()
app.include_router(router)


@app.on_event("startup")
def startup_db_client():
    db = Database()
    app.db = db


@app.on_event("shutdown")
def shutdown_db_client():
    pass


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)

import uvicorn
from fastapi import FastAPI
from app.api.routes import router
from app.state import app_state

app = FastAPI()
app.include_router(router)


@app.on_event("startup")
def startup_db_client():
    # TODO: import globally
    app.state = app_state


@app.on_event("shutdown")
def shutdown_db_client():
    pass


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)

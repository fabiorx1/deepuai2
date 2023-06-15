from fastapi import FastAPI
from .metadata import metadata
from .routes import apps, datasets

app = FastAPI()

@app.get("/")
async def root():
    return {'repository': metadata}

app.include_router(apps.router)
app.include_router(datasets.router)
app.mount('/datasets', datasets.statics)
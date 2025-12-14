import pandas as pd
import json
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import get_full_data
from contextlib import asynccontextmanager


class Search_Query(BaseModel):
    search_string: str | None = None
    min_rating: float | None = None
    max_rating: float | None = None


db = {}


app = FastAPI()


@app.get("/")
def start_server():
    return {"okay": True}


@app.post("/filter", tags=["Products"])
async def search_products(query: Search_Query):
    return "in construction"

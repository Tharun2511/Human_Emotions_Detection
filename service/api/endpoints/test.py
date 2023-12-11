from fastapi import APIRouter, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
import numpy as np

test_router = APIRouter()

@test_router.post("/test/")
async def testing(im: UploadFile):
    return {"testing": "testing"}

from fastapi import FastAPI, UploadFile, File
import shutil
import subprocess
import uuid
import os

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/extract")
async def extract_image(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    input_path = f"{UPLOAD_DIR}/{uid}.png"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Call your existing script
    cmd = [
        "python",
        "system_extract.py",
        "--input", input_path,
        "--output", OUTPUT_DIR,
    ]

    subprocess.run(cmd, check=True)

    return {
        "status": "ok",
        "auth": "tampered",
        "mask": f"/outputs/{uid}_mask.png",
        "recovered": f"/outputs/{uid}_recovered.png"
    }
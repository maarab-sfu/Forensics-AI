import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import torch

# --------------------------------------------------
# Import your wrappers
# --------------------------------------------------
from system_embed import embed_single_image
from system_extract import extract_single_image

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MAX_IMAGE_SIZE_MB = 8
OUTPUT_DIR = "outputs"
PROTECTED_DIR = os.path.join(OUTPUT_DIR, "protected")
RESTORED_DIR = os.path.join(OUTPUT_DIR, "restored")
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")

for d in [PROTECTED_DIR, RESTORED_DIR, MASK_DIR]:
    os.makedirs(d, exist_ok=True)

# --------------------------------------------------
# App initialization
# --------------------------------------------------
app = FastAPI(title="ImageShield Demo API")

# --------------------------------------------------
# CORS (GitHub Pages + local testing)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://maarab-sfu.github.io",
        "http://localhost",
        "http://127.0.0.1",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def validate_image(file: UploadFile):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image type")

    file.file.seek(0, os.SEEK_END)
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)

    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(status_code=400, detail="Image too large")


def load_pil_image(file: UploadFile) -> Image.Image:
    try:
        return Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read image")


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.post("/embed")
async def embed_image(file: UploadFile = File(...)):
    validate_image(file)
    pil_img = load_pil_image(file)

    try:
        protected_img = embed_single_image(pil_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    uid = f"{uuid.uuid4().hex}.png"
    out_path = os.path.join(PROTECTED_DIR, uid)
    protected_img.save(out_path)

    return {
        "protected": f"/outputs/protected/{uid}"
    }


@app.post("/extract")
async def extract_image(file: UploadFile = File(...)):
    validate_image(file)
    pil_img = load_pil_image(file)

    try:
        restored_img, mask_img = extract_single_image(pil_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    uid = uuid.uuid4().hex
    restored_path = os.path.join(RESTORED_DIR, f"{uid}.png")
    mask_path = os.path.join(MASK_DIR, f"{uid}.png")

    restored_img.save(restored_path)
    mask_img.save(mask_path)

    # Simple authentication decision
    mask_np = np.array(mask_img)
    mask_bin = (torch.from_numpy(mask_np).float().mean(dim=-1) > 10).numpy()

    mask_tensor = torch.from_numpy(mask_bin)
    tampered = bool(mask_tensor.sum() > 100)

    return {
        "restored": f"/outputs/restored/{uid}.png",
        "mask": f"/outputs/masks/{uid}.png",
        "auth": "tampered" if tampered else "authentic"
    }


@app.get("/outputs/{subdir}/{filename}")
async def get_output(subdir: str, filename: str):
    path = os.path.join(OUTPUT_DIR, subdir, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)

# backend/preprocess.py
import uuid, json
from pathlib import Path
from PIL import Image

SRC = Path("../data/raw/Skyview")
OUT = Path("../data/processed/skyview")
OUT.mkdir(parents=True, exist_ok=True)

categories = [d.name for d in SRC.iterdir() if d.is_dir()]

manifest = []

for category in categories:
    src_dir = SRC / category
    out_dir = OUT / category
    out_dir.mkdir(parents=True, exist_ok=True)

    for file in src_dir.iterdir():
        if file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        
        img_uuid = str(uuid.uuid4())
        out_path = out_dir / f"{img_uuid}.jpg"

        try:
            img = Image.open(file).convert("RGB")
            img.save(out_path)
        except Exception as e:
            print(f"Corrupted image skipped: {file} – {e}")
            continue

        manifest.append({
            "uuid": img_uuid,
            "filepath": str(out_path.resolve()),
            "category": category
        })

with open(OUT / "manifest.json", "w", encoding="utf8") as f:
    json.dump(manifest, f, indent=2)

print("DONE. Images processed:", len(manifest))

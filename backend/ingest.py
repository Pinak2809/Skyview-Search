# backend/ingest.py
import json
from db import init_db, ImageRecord
from pathlib import Path

Session = init_db()
session = Session()

MANIFEST = Path("../data/processed/skyview/manifest.json")
with MANIFEST.open("r", encoding="utf8") as fh:
    data = json.load(fh)

for item in data:
    if session.query(ImageRecord).filter_by(uuid=item["uuid"]).first():
        continue
    rec = ImageRecord(uuid=item["uuid"], filepath=item["filepath"], category=item["category"])
    session.add(rec)
session.commit()
print("Inserted", len(data))

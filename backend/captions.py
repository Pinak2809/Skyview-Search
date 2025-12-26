# backend/captions.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from db import init_db, ImageRecord

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
model.eval()

def caption_image(path):
    im = Image.open(path).convert("RGB")
    inputs = proc(images=im, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)
    return proc.decode(out[0], skip_special_tokens=True)

def main():
    Session = init_db()
    session = Session()
    
    rows = session.query(ImageRecord).filter(ImageRecord.caption == None).all()
    total = len(rows)
    print(f"Images to caption: {total}")
    
    if total == 0:
        print("Nothing to do.")
        return
    
    BATCH_COMMIT = 100  # commit every 100 images
    
    for i, r in enumerate(rows):
        try:
            cap = caption_image(r.filepath)
            r.caption = cap
            
            # Progress every 100 images
            if (i + 1) % 100 == 0:
                session.commit()
                print(f"Progress: {i + 1}/{total} ({100*(i+1)/total:.1f}%)")
        except Exception as e:
            print(f"Error {r.uuid}: {e}")
    
    # Final commit for remaining
    session.commit()
    print(f"Done. Captioned {total} images.")

if __name__ == "__main__":
    main()
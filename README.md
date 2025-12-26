# Skyview Search

A semantic image search engine for aerial landscape imagery using CLIP embeddings and FAISS vector search.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

Skyview Search enables natural language queries against a dataset of 12,000 aerial images across 15 landscape categories. Instead of relying on keywords or tags, it uses deep learning to understand the visual content of images and match them to text descriptions.

**Example:** Searching "circular irrigation patterns in agricultural land" returns relevant aerial farm images — without any keyword matching.

## Features

- **Semantic Search**: Find images using natural language descriptions
- **Fast Retrieval**: Sub-100ms search across 12,000 images using FAISS
- **REST API**: FastAPI backend with interactive documentation
- **High Accuracy**: 98.3% Recall@5 on evaluation benchmark
- **GPU Accelerated**: CUDA support for fast embedding generation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SKYVIEW SEARCH                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   SQLite    │    │    FAISS    │    │   Mapping   │     │
│  │  (metadata) │    │  (vectors)  │    │ (idx→uuid)  │     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
│         │                  │                  │             │
│         └────────────┬─────┴──────────────────┘             │
│                      │                                      │
│              ┌───────▼───────┐                              │
│              │  search_util  │                              │
│              │  (CLIP model) │                              │
│              └───────┬───────┘                              │
│                      │                                      │
│              ┌───────▼───────┐                              │
│              │    FastAPI    │                              │
│              │   (app.py)    │                              │
│              └───────┬───────┘                              │
│                      │                                      │
│              ┌───────▼───────┐                              │
│              │  HTTP :8000   │                              │
│              └───────────────┘                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## How It Works

### Offline Pipeline (One-time Setup)

1. **Preprocess**: Resize images to 256x256, assign UUIDs
2. **Ingest**: Store metadata in SQLite database
3. **Caption**: Generate descriptions using BLIP model (for display)
4. **Embed**: Convert images to 512-dim vectors using CLIP, store in FAISS

### Online Search (Runtime)

1. User enters text query (e.g., "airport runway")
2. CLIP text encoder converts query to 512-dim vector
3. FAISS finds nearest image vectors (cosine similarity)
4. Return matching images with scores and captions

## Dataset

**Name:** Skyview Multi-Landscape Aerial Imagery Dataset

| Property | Value |
|----------|-------|
| Total Images | 12,000 |
| Categories | 15 |
| Images per Category | 800 |
| Resolution | 256x256 pixels |

### Categories

Agriculture, Airport, Beach, City, Desert, Forest, Grassland, Highway, Lake, Mountain, Parking, Port, Railway, Residential, River

### Sources

- [AID Dataset](https://captain-whu.github.io/AID/)
- [NWPU-RESISC45 Dataset](https://paperswithcode.com/dataset/resisc45)

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/skyview-search.git
   cd skyview-search
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .\.venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r backend/requirements.txt
   ```

4. **Install PyTorch with CUDA (recommended)**
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CPU only
   pip install torch torchvision torchaudio
   ```

5. **Verify GPU availability**
   ```bash
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```

## Dataset Setup

The images are not included in this repository due to size.

1. **Download datasets**
   - [AID Dataset](https://captain-whu.github.io/AID/)
   - [NWPU-RESISC45](https://paperswithcode.com/dataset/resisc45)

2. **Place in data folder**
   ```
   data/
   └── raw/
       ├── AID/
       │   ├── Agriculture/
       │   ├── Airport/
       │   └── ...
       └── NWPU-RESISC45/
           ├── Agriculture/
           ├── Airport/
           └── ...
   ```

3. **Run preprocessing pipeline**
   ```bash
   cd backend
   
   # Step 1: Preprocess images (assign UUIDs, resize)
   python preprocess.py
   
   # Step 2: Ingest to database
   python ingest.py
   
   # Step 3: Generate captions (20-40 min on GPU)
   python captions.py
   
   # Step 4: Generate embeddings (10-20 min on GPU)
   python embed.py
   ```

## Usage

### Start the API Server

```bash
cd backend
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/search?q={query}&k={count}` | GET | Search images by text |
| `/image/{uuid}` | GET | Get full image |
| `/thumbnail/{uuid}` | GET | Get thumbnail |
| `/info/{uuid}` | GET | Get image metadata |
| `/stats` | GET | Database statistics |
| `/categories` | GET | List all categories |

### Interactive Documentation

Open http://127.0.0.1:8000/docs for Swagger UI

### Example Queries

```bash
# Search for airport images
curl "http://127.0.0.1:8000/search?q=airport%20runway&k=5"

# Get database stats
curl "http://127.0.0.1:8000/stats"

# Get image by UUID
curl "http://127.0.0.1:8000/image/{uuid}" --output image.jpg
```

### Python Usage

```python
from search_util import search_text

# Search for images
results = search_text("mountain peaks with snow", k=5)

for r in results:
    print(f"{r['score']:.3f} - {r['caption']}")
```

## Evaluation

### Run Evaluation

```bash
cd backend
python evaluate.py
```

### Results

| Metric | Score |
|--------|-------|
| Recall@1 | 91.7% |
| Recall@5 | 98.3% |
| Recall@10 | 100.0% |
| MRR | 0.952 |

**Target: Recall@5 ≥ 0.6** ✓ Exceeded

### Per-Category Performance

| Category | Recall@5 |
|----------|----------|
| Agriculture | 100% |
| Airport | 100% |
| Beach | 75% |
| City | 100% |
| Desert | 100% |
| Forest | 100% |
| Grassland | 100% |
| Highway | 100% |
| Lake | 100% |
| Mountain | 100% |
| Parking | 100% |
| Port | 100% |
| Railway | 100% |
| Residential | 100% |
| River | 100% |

## Project Structure

```
skyview-search/
├── backend/
│   ├── app.py              # FastAPI REST API
│   ├── captions.py         # BLIP caption generation
│   ├── db.py               # SQLAlchemy database models
│   ├── embed.py            # CLIP embedding generation
│   ├── evaluate.py         # Evaluation harness
│   ├── ingest.py           # Database ingestion
│   ├── preprocess.py       # Image preprocessing
│   ├── search_util.py      # Search functions
│   ├── test_queries.py     # Query testing script
│   ├── test_new_image.py   # Test with new images
│   └── requirements.txt    # Python dependencies
├── data/
│   ├── raw/                # Original datasets (not in repo)
│   └── processed/
│       └── skyview/        # Processed data (not in repo)
├── docs/
├── frontend/               # WPF frontend (planned)
├── .gitignore
└── README.md
```

## Technical Details

### Models Used

| Model | Purpose | Output |
|-------|---------|--------|
| OpenCLIP ViT-B-32 | Image & text embeddings | 512-dim vectors |
| BLIP-base | Image captioning | Text descriptions |

### Database Schema

```sql
CREATE TABLE images (
    uuid TEXT PRIMARY KEY,
    filepath TEXT NOT NULL,
    category TEXT,
    caption TEXT,
    image_metadata JSON,
    embedding_id INTEGER,
    created_at TIMESTAMP
);
```

### Vector Index

- **Type**: FAISS IndexFlatIP
- **Dimensions**: 512
- **Similarity**: Cosine (via inner product on normalized vectors)
- **Size**: ~24MB for 12,000 images

## Known Limitations

1. **No negation understanding**: "no buildings" still matches buildings (CLIP limitation)
2. **Dataset-bound**: Cannot find concepts not in training data
3. **Caption quality varies**: BLIP occasionally produces odd descriptions
4. **English only**: Queries must be in English

## Future Improvements

- [ ] WPF desktop frontend
- [ ] Web-based frontend
- [ ] Image-to-image search
- [ ] Batch upload API
- [ ] Index optimization (IVF+PQ) for larger datasets
- [ ] Multi-language query support

## Dependencies

```
fastapi
uvicorn[standard]
sqlalchemy
pillow
aiofiles
python-multipart
transformers
torch
torchvision
open-clip-torch
faiss-cpu
numpy
```

## References

- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Learning Transferable Visual Models From Natural Language Supervision
- [OpenCLIP](https://github.com/mlfoundations/open_clip) - Open source implementation of CLIP
- [BLIP](https://arxiv.org/abs/2201.12086) - Bootstrapping Language-Image Pre-training
- [FAISS](https://github.com/facebookresearch/faiss) - Library for efficient similarity search

## License

MIT License

## Author

Pinak Ganatra

## Acknowledgments

- AID Dataset contributors
- NWPU-RESISC45 Dataset contributors
- OpenAI (CLIP)
- Salesforce (BLIP)
- Meta (FAISS)

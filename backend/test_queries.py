# backend/test_queries.py
from search_util import search_text
from db import init_db, ImageRecord

Session = init_db()

QUERIES = [
    # Complex descriptive
    "circular irrigation patterns in agricultural land",
    "urban area with tall buildings and dense streets",
    "blue water surrounded by green vegetation",
    
    # Ambiguous (could match multiple categories)
    "vehicles in rows",
    "water and land",
    "green and brown terrain",
    
    # Abstract / conceptual
    "transportation infrastructure",
    "human settlement",
    "natural landscape untouched",
    
    # Negative / unusual
    "no buildings only nature",
    "something random xyz",
    
    # Cross-category
    "boats and water near buildings",
    "road passing through forest",
    "airplane near water",
]

def get_category(uuid):
    session = Session()
    rec = session.query(ImageRecord).filter_by(uuid=uuid).first()
    session.close()
    return rec.category if rec else "Unknown"

def run_tests():
    for query in QUERIES:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        
        results = search_text(query, k=5)
        
        for i, r in enumerate(results, 1):
            category = get_category(r['uuid'])
            caption = r['caption'][:50] if r['caption'] else "No caption"
            print(f"  {i}. {r['score']:.3f} | {category:12} | {caption}")

if __name__ == "__main__":
    run_tests()
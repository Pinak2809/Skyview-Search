# backend/evaluate.py
"""
Evaluation harness for Skyview semantic image search.
Computes Recall@1, Recall@5, Recall@10, and MRR (Mean Reciprocal Rank).

Ground truth: A query like "airport runway" should return images from the "Airport" category.
"""

import json
import numpy as np
from db import init_db, ImageRecord
from search_util import search_text

# Ground truth queries mapped to expected categories
# Each query should retrieve images from the specified category
GROUND_TRUTH_QUERIES = [
    # Agriculture
    {"query": "agricultural fields", "category": "Agriculture"},
    {"query": "farmland crops aerial view", "category": "Agriculture"},
    {"query": "irrigation circles farming", "category": "Agriculture"},
    {"query": "green fields cultivation", "category": "Agriculture"},
    
    # Airport
    {"query": "airport runway", "category": "Airport"},
    {"query": "airplane terminal aerial", "category": "Airport"},
    {"query": "aircraft landing strip", "category": "Airport"},
    {"query": "airport tarmac planes", "category": "Airport"},
    
    # Beach
    {"query": "sandy beach ocean", "category": "Beach"},
    {"query": "coastline waves shore", "category": "Beach"},
    {"query": "beach resort aerial", "category": "Beach"},
    {"query": "seaside sand water", "category": "Beach"},
    
    # City
    {"query": "urban city buildings", "category": "City"},
    {"query": "downtown skyscrapers aerial", "category": "City"},
    {"query": "city streets buildings", "category": "City"},
    {"query": "metropolitan area urban", "category": "City"},
    
    # Desert
    {"query": "desert sand dunes", "category": "Desert"},
    {"query": "arid landscape barren", "category": "Desert"},
    {"query": "sandy desert terrain", "category": "Desert"},
    {"query": "dry desert wasteland", "category": "Desert"},
    
    # Forest
    {"query": "dense forest trees", "category": "Forest"},
    {"query": "woodland green canopy", "category": "Forest"},
    {"query": "forest aerial view trees", "category": "Forest"},
    {"query": "thick forest vegetation", "category": "Forest"},
    
    # Grassland
    {"query": "open grassland plains", "category": "Grassland"},
    {"query": "green meadow grass", "category": "Grassland"},
    {"query": "prairie grassland aerial", "category": "Grassland"},
    {"query": "savanna grass field", "category": "Grassland"},
    
    # Highway
    {"query": "highway road cars", "category": "Highway"},
    {"query": "freeway interchange aerial", "category": "Highway"},
    {"query": "motorway traffic lanes", "category": "Highway"},
    {"query": "road highway asphalt", "category": "Highway"},
    
    # Lake
    {"query": "lake water blue", "category": "Lake"},
    {"query": "freshwater lake aerial", "category": "Lake"},
    {"query": "calm lake water body", "category": "Lake"},
    {"query": "lake surrounded by land", "category": "Lake"},
    
    # Mountain
    {"query": "mountain peaks terrain", "category": "Mountain"},
    {"query": "rocky mountains aerial", "category": "Mountain"},
    {"query": "mountain range hills", "category": "Mountain"},
    {"query": "mountainous terrain elevation", "category": "Mountain"},
    
    # Parking
    {"query": "parking lot cars", "category": "Parking"},
    {"query": "car park aerial view", "category": "Parking"},
    {"query": "parking spaces vehicles", "category": "Parking"},
    {"query": "parking garage lot", "category": "Parking"},
    
    # Port
    {"query": "shipping port harbor", "category": "Port"},
    {"query": "cargo ships dock", "category": "Port"},
    {"query": "port containers cranes", "category": "Port"},
    {"query": "harbor boats marina", "category": "Port"},
    
    # Railway
    {"query": "railway tracks train", "category": "Railway"},
    {"query": "railroad station aerial", "category": "Railway"},
    {"query": "train tracks rail", "category": "Railway"},
    {"query": "railway yard trains", "category": "Railway"},
    
    # Residential
    {"query": "residential houses neighborhood", "category": "Residential"},
    {"query": "suburban homes aerial", "category": "Residential"},
    {"query": "housing development streets", "category": "Residential"},
    {"query": "residential area rooftops", "category": "Residential"},
    
    # River
    {"query": "river water flowing", "category": "River"},
    {"query": "winding river aerial", "category": "River"},
    {"query": "river stream water", "category": "River"},
    {"query": "riverbank water channel", "category": "River"},
]


def get_category_uuids():
    """Build mapping of category -> list of UUIDs."""
    Session = init_db()
    session = Session()
    
    category_map = {}
    rows = session.query(ImageRecord).all()
    
    for r in rows:
        if r.category not in category_map:
            category_map[r.category] = []
        category_map[r.category].append(r.uuid)
    
    session.close()
    return category_map


def evaluate_query(query: str, expected_category: str, category_map: dict, k_values: list = [1, 5, 10]):
    """
    Evaluate a single query.
    
    Returns dict with:
        - hits@k: whether any result in top-k is from expected category
        - reciprocal_rank: 1/rank of first correct result (0 if not in top-max(k))
    """
    max_k = max(k_values)
    results = search_text(query, k=max_k)
    
    expected_uuids = set(category_map.get(expected_category, []))
    
    # Find rank of first correct result
    first_correct_rank = None
    for i, res in enumerate(results):
        if res["uuid"] in expected_uuids:
            first_correct_rank = i + 1  # 1-indexed
            break
    
    # Compute metrics
    metrics = {}
    
    for k in k_values:
        top_k_uuids = {r["uuid"] for r in results[:k]}
        hits = len(top_k_uuids & expected_uuids)
        metrics[f"hit@{k}"] = 1 if hits > 0 else 0
    
    # MRR component
    if first_correct_rank is not None:
        metrics["reciprocal_rank"] = 1.0 / first_correct_rank
    else:
        metrics["reciprocal_rank"] = 0.0
    
    metrics["first_correct_rank"] = first_correct_rank
    
    return metrics


def run_evaluation(k_values: list = [1, 5, 10], verbose: bool = True):
    """
    Run full evaluation on all ground truth queries.
    
    Returns:
        - Per-query results
        - Aggregate metrics (Recall@k, MRR)
    """
    print("Loading category mappings...")
    category_map = get_category_uuids()
    
    print(f"Categories found: {list(category_map.keys())}")
    print(f"Total queries to evaluate: {len(GROUND_TRUTH_QUERIES)}")
    print("-" * 60)
    
    all_results = []
    
    for i, gt in enumerate(GROUND_TRUTH_QUERIES):
        query = gt["query"]
        expected_cat = gt["category"]
        
        metrics = evaluate_query(query, expected_cat, category_map, k_values)
        metrics["query"] = query
        metrics["expected_category"] = expected_cat
        
        all_results.append(metrics)
        
        if verbose:
            status = "✓" if metrics["hit@5"] else "✗"
            rank_str = f"rank={metrics['first_correct_rank']}" if metrics['first_correct_rank'] else "not found"
            print(f"[{i+1:02d}] {status} '{query}' -> {expected_cat} ({rank_str})")
    
    # Aggregate metrics
    print("-" * 60)
    print("AGGREGATE METRICS:")
    
    aggregate = {}
    
    for k in k_values:
        recall_at_k = np.mean([r[f"hit@{k}"] for r in all_results])
        aggregate[f"Recall@{k}"] = recall_at_k
        print(f"  Recall@{k}: {recall_at_k:.3f} ({recall_at_k*100:.1f}%)")
    
    mrr = np.mean([r["reciprocal_rank"] for r in all_results])
    aggregate["MRR"] = mrr
    print(f"  MRR: {mrr:.3f}")
    
    # Per-category breakdown
    print("-" * 60)
    print("PER-CATEGORY RECALL@5:")
    
    categories = set(gt["category"] for gt in GROUND_TRUTH_QUERIES)
    for cat in sorted(categories):
        cat_results = [r for r in all_results if r["expected_category"] == cat]
        cat_recall = np.mean([r["hit@5"] for r in cat_results])
        print(f"  {cat}: {cat_recall:.2f}")
    
    return all_results, aggregate


def save_results(all_results: list, aggregate: dict, output_path: str = "../data/processed/skyview/evaluation_results.json"):
    """Save evaluation results to JSON."""
    output = {
        "aggregate_metrics": aggregate,
        "per_query_results": all_results,
        "num_queries": len(all_results)
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("SKYVIEW SEARCH EVALUATION")
    print("=" * 60)
    
    results, aggregate = run_evaluation(k_values=[1, 5, 10], verbose=True)
    save_results(results, aggregate)
    
    # Final summary
    print("=" * 60)
    print("SUMMARY")
    print(f"  Total queries: {len(results)}")
    print(f"  Recall@1: {aggregate['Recall@1']*100:.1f}%")
    print(f"  Recall@5: {aggregate['Recall@5']*100:.1f}%")
    print(f"  Recall@10: {aggregate['Recall@10']*100:.1f}%")
    print(f"  MRR: {aggregate['MRR']:.3f}")
    print("=" * 60)
    
    # Target check
    if aggregate["Recall@5"] >= 0.6:
        print("✓ TARGET MET: Recall@5 >= 0.6")
    else:
        print(f"✗ TARGET NOT MET: Recall@5 = {aggregate['Recall@5']:.2f} (target: 0.6)")
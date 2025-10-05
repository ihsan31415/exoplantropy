"""Test search function for TOI 700"""
import sys
sys.path.insert(0, '.')

from services.exoplanet_service import search_exoplanet_candidates

# Test search for TOI 700
print("Testing search for 'TOI 700' in TESS dataset...")
result = search_exoplanet_candidates(dataset_key="tess", query="TOI 700")
print("\nResult:")
print(result)
print("\n" + "="*80)

# Test search for just "700"
print("\nTesting search for '700' in TESS dataset...")
result2 = search_exoplanet_candidates(dataset_key="tess", query="700")
print("\nResult:")
print(result2)

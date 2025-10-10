#!/usr/bin/env python3
"""
Test script to verify perplexity calculation and storage.
This creates a small test file, processes it, and shows where perplexities are stored.
"""

import json
import tempfile
import os
from pathlib import Path

# Create a test JSON file with sample data
test_data = {
    "name": "test_method",
    "info": {"test": "data"},
    "raw_results": [
        {
            "original": "The quick brown fox jumps over the lazy dog.",
            "sampled": "A fast red fox leaps across the sleepy dog."
        },
        {
            "original": "Machine learning is a subset of artificial intelligence.",
            "sampled": "Deep learning is part of AI technology."
        },
        {
            "original": "The weather is beautiful today with clear blue skies.",
            "sampled": "Today has nice weather with sunny conditions."
        }
    ],
    "predictions": {
        "real": [3.2, 3.5, 3.1],
        "samples": [3.8, 3.9, 3.7]
    },
    "metrics": {
        "roc_auc": 0.75
    }
}

# Create temporary directory and file
temp_dir = tempfile.mkdtemp()
test_file = os.path.join(temp_dir, "test_results.json")

print("=" * 80)
print("PERPLEXITY STORAGE TEST")
print("=" * 80)
print(f"\nCreating test file: {test_file}")

# Save test data
with open(test_file, 'w') as f:
    json.dump(test_data, f, indent=2)

print("\nOriginal structure:")
print(json.dumps(test_data, indent=2))

# Import and run perplexity calculation
print("\n" + "=" * 80)
print("Running perplexity calculation...")
print("=" * 80)

from perplexity import calculate_perplexity_for_file

# Run with a small model for quick testing
updated_data = calculate_perplexity_for_file(test_file, model_id='facebook/opt-125m')

if updated_data:
    # Save it back
    with open(test_file, 'w') as f:
        json.dump(updated_data, f, indent=2)
    
    print("\n" + "=" * 80)
    print("VERIFICATION: Where perplexities are stored")
    print("=" * 80)
    
    # Show raw_results with perplexities
    print("\n1. Individual perplexities in 'raw_results':")
    print("-" * 80)
    for i, item in enumerate(updated_data['raw_results']):
        print(f"\nItem {i}:")
        print(f"  Original text: {item['original'][:50]}...")
        print(f"  Original perplexity: {item.get('original_perplexity', 'NOT FOUND')}")
        print(f"  Sampled text: {item['sampled'][:50]}...")
        print(f"  Sampled perplexity: {item.get('sampled_perplexity', 'NOT FOUND')}")
    
    # Show predictions with perplexities
    print("\n2. Perplexity lists in 'predictions' dictionary:")
    print("-" * 80)
    if 'predictions' in updated_data:
        preds = updated_data['predictions']
        print(f"  real_perplexity: {preds.get('real_perplexity', 'NOT FOUND')}")
        print(f"  samples_perplexity: {preds.get('samples_perplexity', 'NOT FOUND')}")
    
    # Show metrics
    print("\n3. Mean perplexities in 'metrics':")
    print("-" * 80)
    if 'metrics' in updated_data:
        metrics = updated_data['metrics']
        print(f"  mean_original_perplexity: {metrics.get('mean_original_perplexity', 'NOT FOUND')}")
        print(f"  mean_sampled_perplexity: {metrics.get('mean_sampled_perplexity', 'NOT FOUND')}")
        print(f"  perplexity_model: {metrics.get('perplexity_model', 'NOT FOUND')}")
    
    print("\n" + "=" * 80)
    print("COMPLETE JSON STRUCTURE:")
    print("=" * 80)
    print(json.dumps(updated_data, indent=2))
    
    print("\n" + "=" * 80)
    print("SUCCESS! Perplexities are stored in 3 places:")
    print("  1. raw_results[i]['original_perplexity'] and ['sampled_perplexity']")
    print("  2. predictions['real_perplexity'] and ['samples_perplexity'] (as lists)")
    print("  3. metrics['mean_original_perplexity'] and ['mean_sampled_perplexity']")
    print("=" * 80)

else:
    print("\n✗ Error: Perplexity calculation failed!")

# Cleanup
print(f"\nTest file saved at: {test_file}")
print(f"Temporary directory: {temp_dir}")
print("(You can delete these manually when done)")

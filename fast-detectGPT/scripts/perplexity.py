#!/usr/bin/env python3
"""
Calculate perplexity for original and sampled sequences in JSON result files.

Usage:
    python perplexity.py <folder_path> [--model_id MODEL_ID]

Arguments:
    folder_path: Path to folder containing JSON result files
    --model_id: Model ID for perplexity calculation (default: facebook/opt-125m)
                Other options: 
                - facebook/opt-350m
                - facebook/opt-1.3b
                - EleutherAI/gpt-neo-125M
                - EleutherAI/gpt-neo-1.3B
                - microsoft/phi-2
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from evaluate import load
import numpy as np


def calculate_perplexity_for_file(json_path, model_id='facebook/opt-125m'):
    """
    Calculate perplexity for all sequences in a JSON file and update it.
    
    Args:
        json_path: Path to the JSON file
        model_id: Model ID for perplexity calculation
    
    Returns:
        Dictionary with updated data including perplexities
    """
    print(f"\nProcessing: {json_path}")
    
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check if raw_results exists and has the expected structure
    if 'raw_results' not in data:
        print(f"Warning: No 'raw_results' found in {json_path}")
        return None
    
    raw_results = data['raw_results']
    
    # Extract original and sampled texts
    original_texts = []
    sampled_texts = []
    
    for item in raw_results:
        if 'original' in item:
            original_texts.append(item['original'])
        if 'sampled' in item:
            sampled_texts.append(item['sampled'])
    
    if not original_texts and not sampled_texts:
        print(f"Warning: No text data found in {json_path}")
        return None
    
    print(f"Found {len(original_texts)} original and {len(sampled_texts)} sampled texts")
    
    # Load perplexity metric
    print(f"Loading perplexity metric with model: {model_id}")
    perplexity_metric = load("perplexity", module_type="metric")
    
    # Calculate perplexity for original texts
    original_perplexities = []
    if original_texts:
        print("Calculating perplexity for original texts...")
        # Process in batches to avoid memory issues
        batch_size = 10
        for i in tqdm(range(0, len(original_texts), batch_size)):
            batch = original_texts[i:i+batch_size]
            results = perplexity_metric.compute(
                predictions=batch,
                model_id=model_id,
                add_start_token=True
            )
            # Extract individual perplexities
            if 'perplexities' in results:
                original_perplexities.extend(results['perplexities'])
            else:
                # If individual perplexities not available, use mean for all in batch
                original_perplexities.extend([results['mean_perplexity']] * len(batch))
    
    # Calculate perplexity for sampled texts
    sampled_perplexities = []
    if sampled_texts:
        print("Calculating perplexity for sampled texts...")
        batch_size = 10
        for i in tqdm(range(0, len(sampled_texts), batch_size)):
            batch = sampled_texts[i:i+batch_size]
            results = perplexity_metric.compute(
                predictions=batch,
                model_id=model_id,
                add_start_token=True
            )
            # Extract individual perplexities
            if 'perplexities' in results:
                sampled_perplexities.extend(results['perplexities'])
            else:
                # If individual perplexities not available, use mean for all in batch
                sampled_perplexities.extend([results['mean_perplexity']] * len(batch))
    
    # Add perplexity to each item in raw_results
    print(f"Adding perplexities to {len(raw_results)} items in raw_results...")
    for i, item in enumerate(raw_results):
        if i < len(original_perplexities):
            item['original_perplexity'] = float(original_perplexities[i])
        if i < len(sampled_perplexities):
            item['sampled_perplexity'] = float(sampled_perplexities[i])
    
    # Also add to predictions dictionary if it exists (for compatibility)
    if 'predictions' in data:
        print("Adding perplexities to predictions dictionary...")
        if isinstance(data['predictions'], dict):
            if 'real' in data['predictions'] and original_perplexities:
                data['predictions']['real_perplexity'] = [float(p) for p in original_perplexities]
            if 'samples' in data['predictions'] and sampled_perplexities:
                data['predictions']['samples_perplexity'] = [float(p) for p in sampled_perplexities]
    
    # Calculate mean perplexities
    mean_original_perplexity = float(np.mean(original_perplexities)) if original_perplexities else None
    mean_sampled_perplexity = float(np.mean(sampled_perplexities)) if sampled_perplexities else None
    
    # Add to metrics
    if 'metrics' not in data:
        data['metrics'] = {}
    
    data['metrics']['mean_original_perplexity'] = mean_original_perplexity
    data['metrics']['mean_sampled_perplexity'] = mean_sampled_perplexity
    data['metrics']['perplexity_model'] = model_id
    
    print(f"Mean original perplexity: {mean_original_perplexity:.2f}")
    print(f"Mean sampled perplexity: {mean_sampled_perplexity:.2f}")
    print(f"Stored {len(original_perplexities)} original and {len(sampled_perplexities)} sampled perplexities")
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Calculate perplexity for sequences in JSON result files'
    )
    parser.add_argument(
        'folder_path',
        type=str,
        help='Path to folder containing JSON result files'
    )
    parser.add_argument(
        '--model_id',
        type=str,
        default='facebook/opt-125m',
        help='Model ID for perplexity calculation (default: facebook/opt-125m)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.json',
        help='File pattern to match (default: *.json)'
    )
    
    args = parser.parse_args()
    
    # Get all JSON files in the folder
    folder = Path(args.folder_path)
    if not folder.exists():
        print(f"Error: Folder {args.folder_path} does not exist")
        return
    
    json_files = list(folder.glob(args.pattern))
    
    if not json_files:
        print(f"No files matching pattern '{args.pattern}' found in {args.folder_path}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    print(f"Using model: {args.model_id}")
    print("=" * 80)
    
    # Process each file
    for json_file in json_files:
        try:
            updated_data = calculate_perplexity_for_file(str(json_file), args.model_id)
            
            if updated_data is not None:
                # Save the updated file
                with open(json_file, 'w') as f:
                    json.dump(updated_data, f, indent=2)
                print(f"✓ Updated: {json_file}")
            else:
                print(f"✗ Skipped: {json_file}")
        except Exception as e:
            print(f"✗ Error processing {json_file}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("=" * 80)
    
    print(f"\nCompleted processing {len(json_files)} files")


if __name__ == "__main__":
    main()
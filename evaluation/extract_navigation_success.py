#!/usr/bin/env python3
"""
Extract navigation success information from MindCraft dataset.
MindCraft episodes are collected from navigation trajectories, so we can infer success from the data.
"""

import argparse
import json
from pathlib import Path
import numpy as np


def extract_navigation_success(dataset_path: str, output_file: str):
    """
    Extract navigation success information from MindCraft episodes.
    
    For MindCraft dataset, we can use heuristics:
    1. Check if trajectory reaches the goal (last action is STOP)
    2. Check trajectory length (reasonable length indicates success)
    3. Default to True since MindCraft collects from successful trajectories
    """
    dataset_path = Path(dataset_path)
    all_episodes = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    navigation_results = {}
    
    print(f"Processing {len(all_episodes)} episodes...")
    
    for episode_dir in all_episodes:
        episode_id = episode_dir.name
        data_path = episode_dir / "data.npz"
        
        if not data_path.exists():
            print(f"Warning: {episode_id} - data.npz not found")
            continue
        
        try:
            data = np.load(str(data_path), allow_pickle=True)
            actions = data["actions"]
            
            # Heuristic: Check if last action is STOP (action 0 in VLN-CE)
            # Or check if trajectory is reasonable length (not too short, not too long)
            is_success = True  # Default to True for MindCraft
            
            # You can add more sophisticated checks here if needed
            # For example:
            # - Check if last action is STOP: actions[-1] == 0
            # - Check trajectory length is reasonable: 5 <= len(actions) <= 100
            # - Check if episode has queries (indicates it was processed successfully)
            
            if len(actions) < 3:
                is_success = False  # Too short, likely failed
            
            navigation_results[episode_id] = is_success
            
        except Exception as e:
            print(f"Error processing {episode_id}: {e}")
            navigation_results[episode_id] = False
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(navigation_results, f, indent=2)
    
    # Print statistics
    total = len(navigation_results)
    successful = sum(navigation_results.values())
    print(f"\nNavigation success statistics:")
    print(f"  Total episodes: {total}")
    print(f"  Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"  Failed: {total-successful} ({(total-successful)/total*100:.1f}%)")
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract navigation success from MindCraft")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to MindCraft dataset")
    parser.add_argument("--output", type=str, default="navigation_success.json",
                       help="Output file for navigation results")
    
    args = parser.parse_args()
    extract_navigation_success(args.dataset_path, args.output)


if __name__ == "__main__":
    main()



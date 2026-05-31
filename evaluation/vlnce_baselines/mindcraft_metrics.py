#!/usr/bin/env python3
"""
MindCraft Metrics Evaluation Script
Computes evaluation metrics for MindCraft cognitive queries.

Metrics:
1. QA-Acc: Overall Reasoning Accuracy
2. GCA: Goal-Conditioned Accuracy (requires navigation success labels)
3. CMC: Cognitive Map Consistency
4. SR@WA: Reasoning Failure Impact (requires navigation success labels)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import re

import numpy as np


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.lower().strip()
    # Remove punctuation
    answer = re.sub(r'[^\w\s]', '', answer)
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    return answer


def extract_answer_from_response(response: str, options: List[str], ground_truth: Any) -> str:
    """Extract the actual answer from model response."""
    response_normalized = normalize_answer(response)
    response_upper = response.upper().strip()
    
    # Strategy 1: First check if response exactly matches an option (highest priority)
    if options:
        for option in options:
            option_normalized = normalize_answer(option)
            # Exact match
            if response_normalized == option_normalized:
                return option_normalized
            # Response is just the option with minor variations
            if response_normalized.startswith(option_normalized + ' ') or response_normalized.endswith(' ' + option_normalized):
                return option_normalized
    
    # Strategy 2: Try to extract letter answer (A, B, C, etc.) for short responses
    # Only apply if response is very short (likely a letter answer)
    if options and len(options) >= 2 and len(response_normalized) <= 3:
        # Look for single letter at start of response (most common pattern)
        if len(response_upper) > 0 and response_upper[0] in 'ABCDEFGHIJ':
            letter = response_upper[0]
            idx = ord(letter) - ord('A')
            if 0 <= idx < len(options):
                return normalize_answer(options[idx])
    
    # Strategy 3: Look for letter pattern like "A)", "(A)", "A.", "A:" anywhere in response
    if options and len(options) >= 2:
        # Look for letter pattern with punctuation
        match = re.search(r'[(\[]?([A-J])[)\].:,]', response_upper)
        if match:
            letter = match.group(1)
            idx = ord(letter) - ord('A')
            if 0 <= idx < len(options):
                return normalize_answer(options[idx])
        
        # Look for isolated single letter (but not at start of words)
        # Match patterns like "answer is A" or "choose A"
        match = re.search(r'(?:answer|choose|select|option)\s+([A-J])\b', response_upper)
        if match:
            letter = match.group(1)
            idx = ord(letter) - ord('A')
            if 0 <= idx < len(options):
                return normalize_answer(options[idx])
    
    # Strategy 4: Check if response contains any of the options as substring
    if options:
        for option in options:
            option_normalized = normalize_answer(option)
            if option_normalized in response_normalized:
                return option_normalized
    
    # Strategy 3: For specific question types without explicit options
    # For yes/no questions
    if isinstance(ground_truth, str) and ground_truth.lower() in ['yes', 'no']:
        # Check for yes (but not if "no" is also present indicating negation)
        if 'yes' in response_normalized:
            # Make sure it's not "no, not yes" or similar
            no_pos = response_normalized.find('no')
            yes_pos = response_normalized.find('yes')
            if no_pos == -1 or yes_pos < no_pos:
                return 'yes'
        if 'no' in response_normalized:
            return 'no'
    
    # For left/right questions
    if isinstance(ground_truth, str) and ground_truth.lower() in ['left', 'right']:
        # Be careful with "left" appearing in words like "reflected"
        if re.search(r'\bleft\b', response_normalized):
            return 'left'
        if re.search(r'\bright\b', response_normalized):
            return 'right'
    
    # For before/after questions
    if isinstance(ground_truth, str) and ground_truth.lower() in ['before', 'after', 'at the same time']:
        if 'before' in response_normalized:
            return 'before'
        elif 'after' in response_normalized:
            return 'after'
        elif 'same time' in response_normalized or 'simultaneously' in response_normalized:
            return 'at the same time'
    
    # Strategy 4: For room/location names (like L2.1 self-localization)
    # Return the first word that looks like a room name
    if options and len(options) >= 3:
        # Check if any option is a substring of response (for room names)
        for option in options:
            if len(option) > 3:  # Avoid matching very short words
                if option.lower() in response.lower():
                    return normalize_answer(option)
    
    # Return the normalized response as-is
    return response_normalized


def check_answer_correctness(predicted: str, ground_truth: Any, options: List[str]) -> bool:
    """Check if predicted answer matches ground truth."""
    # Extract answer from response
    extracted = extract_answer_from_response(predicted, options, ground_truth)
    gt_normalized = normalize_answer(str(ground_truth))
    
    # Exact match
    if extracted == gt_normalized:
        return True
    
    # Partial match (for longer answers)
    if gt_normalized in extracted or extracted in gt_normalized:
        return True
    
    return False


def compute_qa_acc(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute Overall Reasoning Accuracy (QA-Acc).
    
    QA-Acc = (1/|Q|) * sum(I(A_q = G_q))
    """
    total_queries = 0
    correct_queries = 0
    
    # Per-query-type accuracy
    query_type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for episode_result in predictions:
        for pred in episode_result.get("predictions", []):
            if "error" in pred:
                continue
            
            total_queries += 1
            query_type = pred["query_type"]
            query_type_stats[query_type]["total"] += 1
            
            # Check correctness
            predicted_answer = pred["predicted_answer"]
            ground_truth = pred["ground_truth_answer"]
            options = pred.get("options", [])
            
            is_correct = check_answer_correctness(predicted_answer, ground_truth, options)
            
            if is_correct:
                correct_queries += 1
                query_type_stats[query_type]["correct"] += 1
    
    # Compute overall accuracy
    qa_acc = correct_queries / total_queries if total_queries > 0 else 0.0
    
    # Compute per-query-type accuracy
    query_type_acc = {}
    for qtype, stats in query_type_stats.items():
        query_type_acc[qtype] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
    
    return {
        "QA-Acc": qa_acc,
        "total_queries": total_queries,
        "correct_queries": correct_queries,
        "query_type_accuracy": query_type_acc,
        "query_type_counts": {k: v["total"] for k, v in query_type_stats.items()},
    }


def compute_gca(predictions: List[Dict[str, Any]], navigation_results: Dict[str, bool]) -> Dict[str, float]:
    """
    Compute Goal-Conditioned Accuracy (GCA).
    Only considers queries from trajectories where navigation was successful.
    
    GCA = (1/|Q_succ|) * sum(I(A_q = G_q)) for q in Q_succ
    
    Args:
        predictions: List of episode predictions
        navigation_results: Dict mapping episode_id to navigation success (True/False)
    """
    total_queries_succ = 0
    correct_queries_succ = 0
    
    for episode_result in predictions:
        episode_id = episode_result["episode_id"]
        
        # Check if navigation was successful
        if episode_id not in navigation_results:
            continue
        
        if not navigation_results[episode_id]:
            continue  # Skip unsuccessful trajectories
        
        for pred in episode_result.get("predictions", []):
            if "error" in pred:
                continue
            
            total_queries_succ += 1
            
            # Check correctness
            predicted_answer = pred["predicted_answer"]
            ground_truth = pred["ground_truth_answer"]
            options = pred.get("options", [])
            
            is_correct = check_answer_correctness(predicted_answer, ground_truth, options)
            
            if is_correct:
                correct_queries_succ += 1
    
    gca = correct_queries_succ / total_queries_succ if total_queries_succ > 0 else 0.0
    
    return {
        "GCA": gca,
        "total_queries_successful_nav": total_queries_succ,
        "correct_queries_successful_nav": correct_queries_succ,
    }


def identify_equivalent_probe_sets(predictions: List[Dict[str, Any]]) -> List[List[Tuple[str, int]]]:
    """
    Identify Equivalent Probe Sets - queries targeting the same spatial fact.
    
    Returns:
        List of probe sets, where each set is a list of (episode_id, query_idx) tuples
    """
    # Strategy 1: Group by query type and metadata similarity
    # For L2.2: queries with same object pairs in same position
    # For L1.1: queries about same object in same location
    
    probe_groups = defaultdict(list)
    
    for episode_result in predictions:
        episode_id = episode_result["episode_id"]
        scene_id = episode_result.get("scene_id", "unknown")
        
        for pred in episode_result.get("predictions", []):
            if "error" in pred:
                continue
            
            query_type = pred["query_type"]
            query_idx = pred["query_idx"]
            metadata = pred.get("metadata", {})
            ground_truth = pred["ground_truth_answer"]
            
            # Create a key that represents the "same fact"
            if query_type == "L2.2_LocalSpatialRelation":
                # Group by scene, objects involved, and spatial relation
                obj_a = metadata.get("object_A_id", "")
                obj_b = metadata.get("object_B_id", "")
                # Sort to handle symmetry (A-B vs B-A)
                key = f"{scene_id}|L2.2|{min(obj_a, obj_b)}-{max(obj_a, obj_b)}|{ground_truth}"
            
            elif query_type == "L1.1_ObjectAttributeRecall":
                # Group by scene, object, and spatial relation
                obj_id = metadata.get("probe_object_id", "")
                key = f"{scene_id}|L1.1|{obj_id}|{ground_truth}"
            
            elif query_type == "L1.2_TemporalRelationRecall":
                # Group by scene, object pair, and temporal order
                obj_a = metadata.get("object_A_id", "")
                obj_b = metadata.get("object_B_id", "")
                key = f"{scene_id}|L1.2|{obj_a}-{obj_b}|{ground_truth}"
            
            elif query_type == "L2.1_SelfLocalization":
                # Group by scene, timestep, and correct room
                timestep = pred.get("query_timestep", 0)
                correct_room = metadata.get("correct_room", "")
                # Allow some timestep tolerance (±2 steps)
                timestep_bin = timestep // 3 * 3
                key = f"{scene_id}|L2.1|{timestep_bin}|{correct_room}"
            
            elif query_type == "L3.1_TopologicalAdjacency":
                # Group by scene, query room, and correct adjacent room
                query_room = metadata.get("query_room", "")
                correct_room = metadata.get("correct_answer_room", "")
                key = f"{scene_id}|L3.1|{query_room}|{correct_room}"
            
            elif query_type == "L3.2_LandmarkBasedFuturePathValidation":
                # Group by scene, landmark, and ground truth
                landmark = metadata.get("landmark_room", "")
                key = f"{scene_id}|L3.2|{landmark}|{ground_truth}"
            
            else:
                # Default: group by query type and ground truth
                key = f"{scene_id}|{query_type}|{ground_truth}"
            
            probe_groups[key].append((episode_id, query_idx, pred))
    
    # Filter to only keep groups with at least 2 queries
    equivalent_sets = []
    for key, queries in probe_groups.items():
        if len(queries) >= 2:
            equivalent_sets.append([(ep_id, q_idx) for ep_id, q_idx, _ in queries])
    
    return equivalent_sets


def compute_cmc(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute Cognitive Map Consistency (CMC).
    Measures stability of agent's internal representation.
    
    CMC = C_total / P_total
    where C_total = sum of consistent answer pairs
          P_total = sum of all pairwise comparisons
    """
    # Build answer lookup: (episode_id, query_idx) -> predicted_answer
    answer_lookup = {}
    
    for episode_result in predictions:
        episode_id = episode_result["episode_id"]
        for pred in episode_result.get("predictions", []):
            if "error" in pred:
                continue
            query_idx = pred["query_idx"]
            predicted_answer = pred["predicted_answer"]
            options = pred.get("options", [])
            ground_truth = pred["ground_truth_answer"]
            
            # Extract normalized answer
            extracted = extract_answer_from_response(predicted_answer, options, ground_truth)
            answer_lookup[(episode_id, query_idx)] = extracted
    
    # Identify equivalent probe sets
    equivalent_sets = identify_equivalent_probe_sets(predictions)
    
    print(f"Found {len(equivalent_sets)} equivalent probe sets")
    
    # Compute consistency
    c_total = 0  # Total consistent pairs
    p_total = 0  # Total pairwise comparisons
    
    for probe_set in equivalent_sets:
        k = len(probe_set)
        if k < 2:
            continue
        
        # Count pairwise comparisons: C(k, 2) = k*(k-1)/2
        num_pairs = k * (k - 1) // 2
        p_total += num_pairs
        
        # Count consistent pairs
        for i in range(k):
            for j in range(i + 1, k):
                ep_id_i, q_idx_i = probe_set[i]
                ep_id_j, q_idx_j = probe_set[j]
                
                answer_i = answer_lookup.get((ep_id_i, q_idx_i), "")
                answer_j = answer_lookup.get((ep_id_j, q_idx_j), "")
                
                if answer_i and answer_j and answer_i == answer_j:
                    c_total += 1
    
    cmc = c_total / p_total if p_total > 0 else 0.0
    
    return {
        "CMC": cmc,
        "total_consistent_pairs": c_total,
        "total_pairwise_comparisons": p_total,
        "num_equivalent_probe_sets": len(equivalent_sets),
    }


def compute_sr_at_wa(predictions: List[Dict[str, Any]], navigation_results: Dict[str, bool]) -> Dict[str, float]:
    """
    Compute Reasoning Failure Impact (SR@WA).
    Navigation Success Rate only on trajectories with at least one incorrect answer.
    
    SR@WA = (1/|T_WA|) * sum(S(t)) for t in T_WA
    where T_WA = trajectories with at least one wrong answer
    """
    # Identify trajectories with wrong answers
    trajectories_with_wrong_answer = set()
    
    for episode_result in predictions:
        episode_id = episode_result["episode_id"]
        has_wrong_answer = False
        
        for pred in episode_result.get("predictions", []):
            if "error" in pred:
                continue
            
            predicted_answer = pred["predicted_answer"]
            ground_truth = pred["ground_truth_answer"]
            options = pred.get("options", [])
            
            is_correct = check_answer_correctness(predicted_answer, ground_truth, options)
            
            if not is_correct:
                has_wrong_answer = True
                break
        
        if has_wrong_answer:
            trajectories_with_wrong_answer.add(episode_id)
    
    # Compute success rate among trajectories with wrong answers
    total_trajectories_wa = 0
    successful_trajectories_wa = 0
    
    for episode_id in trajectories_with_wrong_answer:
        if episode_id not in navigation_results:
            continue
        
        total_trajectories_wa += 1
        if navigation_results[episode_id]:
            successful_trajectories_wa += 1
    
    sr_at_wa = successful_trajectories_wa / total_trajectories_wa if total_trajectories_wa > 0 else 0.0
    
    return {
        "SR@WA": sr_at_wa,
        "total_trajectories_with_wrong_answer": total_trajectories_wa,
        "successful_trajectories_with_wrong_answer": successful_trajectories_wa,
    }


def load_predictions(predictions_dir: str) -> List[Dict[str, Any]]:
    """Load all prediction files from directory."""
    predictions_dir = Path(predictions_dir)
    all_predictions = []
    
    for pred_file in sorted(predictions_dir.glob("predictions_chunk_*.json")):
        with open(pred_file, "r") as f:
            chunk_predictions = json.load(f)
            all_predictions.extend(chunk_predictions)
    
    return all_predictions


def load_navigation_results(navigation_results_path: str) -> Dict[str, bool]:
    """Load navigation success results if available."""
    if not navigation_results_path or not os.path.exists(navigation_results_path):
        return {}
    
    with open(navigation_results_path, "r") as f:
        nav_results = json.load(f)
    
    # Convert to dict: episode_id -> success (bool)
    return {str(k): bool(v) for k, v in nav_results.items()}


def main():
    parser = argparse.ArgumentParser(description="MindCraft Metrics Evaluation")
    parser.add_argument("--predictions-dir", type=str, required=True, 
                       help="Directory containing prediction files")
    parser.add_argument("--dataset-path", type=str, required=True, 
                       help="Path to MindCraft dataset (for reference)")
    parser.add_argument("--output-dir", type=str, required=True, 
                       help="Output directory for metrics")
    parser.add_argument("--navigation-results", type=str, default=None,
                       help="Path to navigation success results (optional, for GCA and SR@WA)")
    parser.add_argument("--metrics", type=str, default="all",
                       help="Comma-separated list of metrics to compute (qa-acc,gca,cmc,sr-wa,all)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load predictions
    print("Loading predictions...")
    predictions = load_predictions(args.predictions_dir)
    print(f"Loaded predictions for {len(predictions)} episodes")
    
    # Load navigation results (optional)
    navigation_results = load_navigation_results(args.navigation_results)
    if navigation_results:
        print(f"Loaded navigation results for {len(navigation_results)} episodes")
    else:
        print("No navigation results provided. GCA and SR@WA will be skipped.")
    
    # Parse metrics to compute
    metrics_to_compute = args.metrics.lower().split(",")
    if "all" in metrics_to_compute:
        metrics_to_compute = ["qa-acc", "gca", "cmc", "sr-wa"]
    
    # Compute metrics
    results = {
        "num_episodes": len(predictions),
        "total_queries": sum(len(ep.get("predictions", [])) for ep in predictions),
    }
    
    # 1. QA-Acc
    if "qa-acc" in metrics_to_compute:
        print("\nComputing QA-Acc (Overall Reasoning Accuracy)...")
        qa_acc_results = compute_qa_acc(predictions)
        results.update(qa_acc_results)
        print(f"QA-Acc: {qa_acc_results['QA-Acc']:.4f} ({qa_acc_results['correct_queries']}/{qa_acc_results['total_queries']})")
    
    # 2. GCA
    if "gca" in metrics_to_compute and navigation_results:
        print("\nComputing GCA (Goal-Conditioned Accuracy)...")
        gca_results = compute_gca(predictions, navigation_results)
        results.update(gca_results)
        print(f"GCA: {gca_results['GCA']:.4f} ({gca_results['correct_queries_successful_nav']}/{gca_results['total_queries_successful_nav']})")
    elif "gca" in metrics_to_compute:
        print("\nSkipping GCA (navigation results not provided)")
    
    # 3. CMC
    if "cmc" in metrics_to_compute:
        print("\nComputing CMC (Cognitive Map Consistency)...")
        cmc_results = compute_cmc(predictions)
        results.update(cmc_results)
        print(f"CMC: {cmc_results['CMC']:.4f} ({cmc_results['total_consistent_pairs']}/{cmc_results['total_pairwise_comparisons']})")
    
    # 4. SR@WA
    if "sr-wa" in metrics_to_compute and navigation_results:
        print("\nComputing SR@WA (Reasoning Failure Impact)...")
        sr_wa_results = compute_sr_at_wa(predictions, navigation_results)
        results.update(sr_wa_results)
        print(f"SR@WA: {sr_wa_results['SR@WA']:.4f} ({sr_wa_results['successful_trajectories_with_wrong_answer']}/{sr_wa_results['total_trajectories_with_wrong_answer']})")
    elif "sr-wa" in metrics_to_compute:
        print("\nSkipping SR@WA (navigation results not provided)")
    
    # Save results
    output_file = os.path.join(args.output_dir, "metrics_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Metrics computation completed!")
    print(f"Results saved to {output_file}")
    print(f"{'='*50}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    if "QA-Acc" in results:
        print(f"QA-Acc: {results['QA-Acc']:.4f}")
    if "GCA" in results:
        print(f"GCA: {results['GCA']:.4f}")
    if "CMC" in results:
        print(f"CMC: {results['CMC']:.4f}")
    if "SR@WA" in results:
        print(f"SR@WA: {results['SR@WA']:.4f}")


if __name__ == "__main__":
    main()


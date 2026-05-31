#!/usr/bin/env python3
"""
MindCraft Inference Script
Loads model and generates predictions for MindCraft cognitive queries.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add parent directory to path to import llava
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates


def sample_and_pad_images(images, num_frames=8, width=512, height=512):
    """Sample and pad images to fixed number of frames."""
    from PIL import Image as PILImage
    
    if len(images) == 0:
        return []
    
    if len(images) <= num_frames:
        # Pad with the last frame
        latest_frame = images[-1]
        return images + [latest_frame] * (num_frames - len(images))
    
    # Sample uniformly
    frames = images[:-1]  # All except the last frame
    latest_frame = images[-1]  # Keep the last frame
    
    sampled_indices = np.linspace(0, len(frames) - 1, num_frames - 1, dtype=int)
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]
    
    return sampled_frames


def load_mindcraft_episode(episode_dir: Path) -> Dict[str, Any]:
    """Load a single MindCraft episode from directory."""
    data_path = episode_dir / "data.npz"
    if not data_path.exists():
        return None
    
    data = np.load(str(data_path), allow_pickle=True)
    
    # Extract instruction text
    instruction = data["instruction"]
    if hasattr(instruction, 'item'):
        instruction_obj = instruction.item()
        instruction_text = instruction_obj.instruction_text if hasattr(instruction_obj, 'instruction_text') else str(instruction_obj)
    else:
        instruction_text = str(instruction)
    
    # Extract queries
    queries = data["mindcraft_queries"]
    if hasattr(queries, 'tolist'):
        queries_list = queries.tolist()
    else:
        queries_list = list(queries)
    
    # Load memory_log if possible (optional, not needed for inference)
    try:
        memory_log = data["memory_log"].item() if "memory_log" in data else None
    except:
        memory_log = None
    
    return {
        "episode_id": episode_dir.name,
        "observations": data["observations"],  # [T, H, W, 3]
        "actions": data["actions"],  # [T]
        "instruction": instruction_text,
        "scene_id": str(data["scene_id"]) if "scene_id" in data else "unknown",
        "memory_log": memory_log,
        "mindcraft_queries": queries_list,
        "primary_query": data.get("primary_query", None),
        "semantic_observations": data.get("semantic_observations", None),
    }


def generate_query_answer(
    model,
    tokenizer,
    image_processor,
    query: Dict[str, Any],
    observations: np.ndarray,
    instruction: str,
    conv_mode: str = "llama_3",
    num_frames: int = 8,
) -> str:
    """Generate answer for a single query using the model."""
    
    # Get the query timestep
    query_timestep = query["query_timestep"]
    
    # Get observations up to and including the query timestep
    obs_upto_query = observations[: query_timestep + 1]
    
    # Convert observations to PIL images
    images = []
    for obs in obs_upto_query:
        # Observations are in RGB format [H, W, 3]
        img = Image.fromarray(np.uint8(obs)).convert("RGB")
        images.append(img)
    
    # Sample and pad images
    sampled_images = sample_and_pad_images(images, num_frames=num_frames)
    
    # Prepare the question with instruction and query
    question_text = query["question"]
    
    # Format based on query type
    if query["query_type"].startswith("L1.1") or query["query_type"].startswith("L1.2"):
        # Memory-based queries
        interleaved_images = "<image>\n" * (len(sampled_images) - 1)
        question = (
            f"You are a robot that has been navigating through an environment. "
            f'Your task was: "{instruction}"\n\n'
            f"You have completed your navigation. Here is a video of your path: "
            f"{interleaved_images}and final observation <image>\n\n"
            f"Question: {question_text}\n\n"
        )
    elif query["query_type"].startswith("L2"):
        # Local perception queries
        interleaved_images = "<image>\n" * (len(sampled_images) - 1)
        question = (
            f"You are a robot navigating through an environment. "
            f'Your task is: "{instruction}"\n\n'
            f"Here are your recent observations: "
            f"{interleaved_images}and current observation <image>\n\n"
            f"Question: {question_text}\n\n"
        )
    elif query["query_type"].startswith("L3"):
        # Global reasoning queries
        interleaved_images = "<image>\n" * (len(sampled_images) - 1)
        question = (
            f"You are a robot navigating through an environment. "
            f'Your task is: "{instruction}"\n\n'
            f"Here is your navigation history: "
            f"{interleaved_images}and current observation <image>\n\n"
            f"Question: {question_text}\n\n"
        )
    else:
        # Default format
        interleaved_images = "<image>\n" * (len(sampled_images) - 1)
        question = (
            f'Task: "{instruction}"\n\n'
            f"Observations: {interleaved_images}<image>\n\n"
            f"Question: {question_text}\n\n"
        )
    
    # Add options if this is a multiple choice question
    if query.get("options") and len(query["options"]) > 2:
        options_text = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(query["options"])])
        question += f"Options:\n{options_text}\n\nAnswer with just the letter (A, B, C, etc.) or the answer directly."
    else:
        question += "Please provide a concise answer."
    
    # Create conversation
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Process images
    images_tensor = process_images(sampled_images, image_processor, model.config)
    
    # Get model device (handle multi-device models)
    model_device = getattr(model, 'device', None)
    if model_device is None:
        # Try to get device from first parameter
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    images_tensor = images_tensor.to(model_device, dtype=torch.float16)
    
    # Tokenize
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(model_device)
    
    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=128,  # Reduced from 512 to save memory
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode output
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    # Clear tensors to free memory
    del input_ids, images_tensor, output_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return outputs


def process_episode(
    episode_data: Dict[str, Any],
    model,
    tokenizer,
    image_processor,
    conv_mode: str,
    num_frames: int,
) -> Dict[str, Any]:
    """Process a single episode and generate predictions for all queries."""
    
    episode_id = episode_data["episode_id"]
    observations = episode_data["observations"]
    instruction = episode_data["instruction"]
    queries = episode_data["mindcraft_queries"]
    
    results = {
        "episode_id": episode_id,
        "instruction": instruction,
        "scene_id": episode_data["scene_id"],
        "num_queries": len(queries),
        "predictions": [],
    }
    
    for query_idx, query in enumerate(queries):
        try:
            # Clear CUDA cache before each query to avoid OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Generate answer
            predicted_answer = generate_query_answer(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                query=query,
                observations=observations,
                instruction=instruction,
                conv_mode=conv_mode,
                num_frames=num_frames,
            )
            
            # Store prediction
            prediction = {
                "query_idx": query_idx,
                "query_type": query["query_type"],
                "query_timestep": query["query_timestep"],
                "question": query["question"],
                "predicted_answer": predicted_answer,
                "ground_truth_answer": query["answer"],
                "options": query.get("options", []),
                "metadata": query.get("metadata", {}),
            }
            
            results["predictions"].append(prediction)
            
        except Exception as e:
            print(f"Error processing query {query_idx} in episode {episode_id}: {e}")
            # Clear cache on error too
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            results["predictions"].append({
                "query_idx": query_idx,
                "query_type": query["query_type"],
                "error": str(e),
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="MindCraft Inference")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to MindCraft dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for predictions")
    parser.add_argument("--num-chunks", type=int, default=1, help="Number of chunks to split dataset")
    parser.add_argument("--chunk-idx", type=int, default=0, help="Index of current chunk")
    parser.add_argument("--conv-mode", type=str, default="llama_3", help="Conversation mode")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to sample")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model_name = os.path.basename(os.path.normpath(args.model_path))
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, model_name
    )
    
    # Move model to GPU (handle offloaded parameters)
    if hasattr(model, 'model') and hasattr(model.model, 'vision_tower'):
        # Model might already be on GPU with device_map
        print("Model loaded with device_map, checking device placement...")
    else:
        try:
            model = model.cuda()
        except (NotImplementedError, RuntimeError) as e:
            print(f"Note: Could not move all parameters to CUDA: {e}")
            print("Model may be using device_map for distributed placement")
    
    model.eval()
    print("Model loaded successfully!")
    
    # Get all episode directories
    dataset_path = Path(args.dataset_path)
    all_episodes = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    # Split episodes into chunks
    chunk_size = len(all_episodes) // args.num_chunks
    start_idx = args.chunk_idx * chunk_size
    end_idx = start_idx + chunk_size if args.chunk_idx < args.num_chunks - 1 else len(all_episodes)
    episodes_to_process = all_episodes[start_idx:end_idx]
    
    print(f"Processing chunk {args.chunk_idx + 1}/{args.num_chunks}")
    print(f"Episodes: {start_idx} to {end_idx} (total: {len(episodes_to_process)})")
    
    # Process episodes
    all_results = []
    
    for episode_dir in tqdm(episodes_to_process, desc="Processing episodes"):
        # Load episode
        episode_data = load_mindcraft_episode(episode_dir)
        if episode_data is None:
            print(f"Warning: Could not load episode {episode_dir.name}")
            continue
        
        # Process episode
        results = process_episode(
            episode_data=episode_data,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            conv_mode=args.conv_mode,
            num_frames=args.num_frames,
        )
        
        all_results.append(results)
    
    # Save results
    output_file = os.path.join(args.output_dir, f"predictions_chunk_{args.chunk_idx}.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nInference completed!")
    print(f"Processed {len(all_results)} episodes")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()


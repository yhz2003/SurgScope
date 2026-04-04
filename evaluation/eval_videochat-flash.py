import os
import json
import torch
import numpy as np
import re
from typing import List, Tuple, Optional, Set, Dict, Any
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import warnings

# Ignore unnecessary warnings for a cleaner console output
warnings.filterwarnings("ignore", category=UserWarning)

# ================= Configuration =================
# Model weights path
MODEL_PATH = ""

# Dataset JSON path containing background knowledge
DATASET_JSON_PATH = "" 
# Output log file path (indicates full video evaluation with context)
OUTPUT_LOG_FILE = ""

# Dataset root directory mapping
DATA_ROOTS = {
    'pitvis':     '',
    'm2cai16':    '',
    'mvp57':      '',
    'autolaparo': '',
    'cataract':   '',
    'case':       ''
}

# VRAM optimization: Uniformly sampled frame count for full video
MAX_NUM_FRAMES = 100

# ================= Utility Functions =================

def resolve_video_path(item_id: str) -> Optional[str]:
    """
    Resolves the absolute video path based on the dataset prefix and item ID.
    Handles both .mp4 and .avi extensions.
    """
    try:
        parts = str(item_id).lower().split('_')
        prefix = parts[0]
        vid_str = parts[1]
        
        folder_path = DATA_ROOTS.get(prefix)
        if not folder_path: 
            return None
            
        vid_num = int(vid_str)
        p1 = os.path.join(folder_path, f"{vid_num}.mp4")
        if os.path.exists(p1): 
            return p1
            
        p2 = os.path.join(folder_path, f"{vid_num}.avi")
        if os.path.exists(p2): 
            return p2
            
        return p1  # Default to mp4 if neither is found immediately
    except (IndexError, KeyError, ValueError):
        return None

def calculate_iou(pred: List[float], gt: List[float]) -> float:
    """Calculates the Temporal Intersection over Union (IoU)."""
    start_pred, end_pred = pred
    start_gt, end_gt = gt
    
    if start_pred > end_pred: 
        start_pred, end_pred = end_pred, start_pred
        
    intersection = max(0, min(end_pred, end_gt) - max(start_pred, start_gt))
    union = (end_pred - start_pred) + (end_gt - start_gt) - intersection
    
    if union <= 0: 
        return 0.0
    return intersection / union

def calculate_max_iou(pred: List[float], gt_list: List[List[float]]) -> float:
    """Calculates the maximum IoU against a list of ground truth segments."""
    if not gt_list: 
        return 0.0
    ious = [calculate_iou(pred, gt) for gt in gt_list]
    return max(ious)

def parse_model_response(response: str) -> List[float]:
    """
    Parses the model's text response to extract start and end timestamps.
    """
    try:
        response = response.strip()
        # Priority match: Brackets containing timestamps
        if '[' in response and ']' in response:
            content = response[response.find('[')+1 : response.find(']')]
            parts = content.split(',')
            if len(parts) >= 2:
                return [float(parts[0].strip()), float(parts[1].strip())]
        
        # Fallback match: Extract the last two consecutive numbers
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        if len(nums) >= 2:
            return [float(nums[-2]), float(nums[-1])]
    except Exception:
        pass
    return [0.0, 0.0]

# ================= Metrics Aggregation & Checkpointing =================

def get_processed_ids(log_file: str) -> Set[str]:
    """
    Scans the log file to resume execution from the last breakpoint.
    """
    processed = set()
    if os.path.exists(log_file):
        print(f"🔄 Scanning checkpoint file: {log_file}")
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split('\t')
                if len(parts) > 0 and parts[0] != "ID" and not parts[0].startswith("==="):
                    processed.add(parts[0])
    return processed

def write_final_report(log_file: str, missing_files: List[str]) -> None:
    """
    Calculates and appends the final global evaluation metrics to the log file.
    """
    all_ious = []
    missing_count = 0
    error_count = 0
    
    if not os.path.exists(log_file):
        return

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("===") or line.startswith("ID\t"):
                continue
            
            parts = line.split('\t')
            if len(parts) >= 5:
                if 'MISSING' in line:
                    missing_count += 1
                    all_ious.append(0.0)
                elif 'ERROR' in line:
                    error_count += 1
                    all_ious.append(0.0)
                elif 'NO_GT' in line:
                    pass 
                else:
                    try:
                        # Assuming IoU is at index 4 based on the standardized header
                        iou = float(parts[4])
                        all_ious.append(iou)
                    except ValueError:
                        pass

    total = len(all_ious)
    if total == 0:
        print("No valid samples processed to generate a report.")
        return

    r1_03 = sum(1 for x in all_ious if x >= 0.3)
    r1_05 = sum(1 for x in all_ious if x >= 0.5)
    r1_07 = sum(1 for x in all_ious if x >= 0.7)

    metrics_str = (
        f"\n================ VideoChat-Flash Global Eval Report (Full Video + Context) ================\n"
        f"Total Evaluated Samples (incl. checkpoints) : {total}\n"
        f"Missing Files (MISSING)                     : {missing_count}\n"
        f"Exceptions/Errors (ERROR)                   : {error_count}\n"
        f"mIoU                                        : {(np.mean(all_ious) * 100):.2f}%\n"
        f"R@1 (IoU >= 0.3)                            : {(r1_03 / total * 100):.2f}%\n"
        f"R@1 (IoU >= 0.5)                            : {(r1_05 / total * 100):.2f}%\n"
        f"R@1 (IoU >= 0.7)                            : {(r1_07 / total * 100):.2f}%\n"
        f"===========================================================================================\n"
    )
    
    print(metrics_str)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(metrics_str)
        if missing_files:
            f.write("\n\n=== Missing Files Summary ===\n")
            f.write("\n".join(missing_files))
            f.write("\n==============================\n")

# ================= Main Execution Flow =================

def main():
    # 0. Create log directory
    if OUTPUT_LOG_FILE:
        log_dir = os.path.dirname(OUTPUT_LOG_FILE)
        if log_dir and not os.path.exists(log_dir): 
            os.makedirs(log_dir, exist_ok=True)

    # 1. Load Model
    print(f"📦 Loading Model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).cuda()
    model.eval()

    # 2. Load Dataset
    print(f"📖 Loading Dataset: {DATASET_JSON_PATH}")
    with open(DATASET_JSON_PATH, 'r') as f:
        data = json.load(f)
        
    if isinstance(data, dict):
        if 'dataset' in data: 
            data = data['dataset']
        elif 'data' in data: 
            data = data['data']
            
    print(f"Total samples to evaluate: {len(data)}")

    # 3. Setup Logging & Checkpoint Resumption
    processed_ids = get_processed_ids(OUTPUT_LOG_FILE)
    print(f"📌 Detected processed samples: {len(processed_ids)}")
    
    write_mode = 'a' if len(processed_ids) > 0 else 'w'
    need_header = len(processed_ids) == 0

    ious = []
    missing_files = []

    print(f"🚀 Start Inference (Complete Video + Background Context) -> {OUTPUT_LOG_FILE}")
    with open(OUTPUT_LOG_FILE, write_mode, encoding='utf-8') as f_log:
        if need_header:
            # Standardized 7-column header to match previous scripts
            f_log.write("ID\tPhase\tPred_Abs\tGT_Abs\tIoU\tRawResponse\tPromptMsg\n")
            f_log.flush()
        
        for item in tqdm(data, desc="Evaluating Full Video"):
            item_id = item.get('id', 'unknown')
            phase = item.get('phase', '')
            
            # Checkpoint skipping
            if item_id in processed_ids:
                continue
            
            # --- Robust Path Resolution ---
            video_path = resolve_video_path(item_id)
            
            # File validation
            if not video_path or not os.path.exists(video_path):
                missing_files.append(f"{item_id} -> {video_path}")
                ious.append(0.0)
                f_log.write(f"{item_id}\t{phase}\t[0,0]\t[]\t0.0\tMISSING_FILE\tMISSING\n")
                f_log.flush()
                continue

            # Extract Ground Truth (Absolute time for full video prediction)
            gt_segments = []
            if 'time' in item:
                for t in item['time']:
                    gt_segments.append([float(t['s0']), float(t['e0'])])
            
            if not gt_segments:
                f_log.write(f"{item_id}\t{phase}\t[0,0]\t[]\t0.0\tNO_GT\tNO_GT\n")
                f_log.flush()
                continue

            # --- Extract Background Knowledge ---
            conv_data = item.get('conversation', [{}])[0]
            query_text = conv_data.get('query1', f"Identify the start and end timestamps for the phase: {phase}")
            objective = conv_data.get('objective', '')
            anatomy = conv_data.get('key anatomical structures', '')
            steps = conv_data.get('key steps', '')

            bg_text = ""
            if objective or anatomy or steps:
                bg_text += "Background Knowledge:\n"
                if objective: bg_text += f"- Objective: {objective}\n"
                if anatomy: bg_text += f"- Anatomy: {anatomy}\n"
                if steps: bg_text += f"- Key Steps: {steps}\n\n"

            # --- Construct Prompt (Full Video Version) ---
            prompt = (
                f"You are a surgical assistant. Analyze the surgical video provided.\n"
                f"{bg_text}"
                f"Question: {query_text}\n"
                f"Format your answer strictly as: [start, end]"
            )

            try:
                # Increased max_new_tokens to 128 to prevent truncation with context
                gen_config = {"max_new_tokens": 128, "do_sample": False}
                
                # Direct full video inference (no ffmpeg slicing needed)
                response, _ = model.chat(
                    video_path=video_path,
                    tokenizer=tokenizer,
                    user_prompt=prompt,
                    return_history=True,
                    max_num_frames=MAX_NUM_FRAMES,  # Model samples uniformly
                    generation_config=gen_config
                )
                
                # Sanitize newlines/tabs to maintain log format integrity
                response_clean = response.replace('\n', ' ').replace('\t', ' ').strip()
                prompt_clean = prompt.replace('\n', ' ').replace('\t', ' ').strip()

                # Parse model predictions
                pred_timestamp = parse_model_response(response)
                
                # Calculate IoU
                max_iou = calculate_max_iou(pred_timestamp, gt_segments)
                ious.append(max_iou)
                
                # Write to log (Standardized formatting)
                f_log.write(f"{item_id}\t{phase}\t{pred_timestamp}\t{gt_segments}\t{max_iou:.4f}\t{response_clean}\t{prompt_clean}\n")
                f_log.flush()
                
            except Exception as e:
                f_log.write(f"{item_id}\t{phase}\t[0,0]\t[]\t0.0\tERROR: {str(e)}\tERROR\n")
                f_log.flush()
                ious.append(0.0)

    # 4. Generate Final Evaluation Report
    write_final_report(OUTPUT_LOG_FILE, missing_files)

if __name__ == "__main__":
    main()
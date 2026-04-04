import os
import json
import torch
import numpy as np
import re
from typing import List, Tuple, Optional, Set, Dict, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
import warnings

# Ignore unnecessary warnings for cleaner console output
warnings.filterwarnings("ignore")

# ================= Configuration =================
# Model weights path
MODEL_PATH = ""

# Dataset JSON path containing background knowledge
DATASET_JSON_PATH = ""

# Output log file path (e.g., phase_full_bg.txt for comparison)
OUTPUT_LOG_FILE = ""

# Dataset root directory mapping
DATA_ROOTS = {
    'pitvis':     '',
    'm2cai16':    '',
    'mvp57':      '',
    'autolaparo': '',
    'cataract':   ''
}

# Strict alignment with baseline video sampling parameters
FPS = 1
MAX_FRAMES = 100

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
        
        # Check compatibility for both formats
        p1 = os.path.join(folder_path, f"{vid_num}.mp4")
        if os.path.exists(p1): 
            return p1
            
        p2 = os.path.join(folder_path, f"{vid_num}.avi")
        if os.path.exists(p2): 
            return p2
            
        return p1  # Default fallback
    except Exception:
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
                s = float(re.sub(r"[^0-9.]", "", parts[0]))
                e = float(re.sub(r"[^0-9.]", "", parts[1]))
                return [s, e]
                
        # Fallback match: Extract the first two valid numbers
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        nums = [float(n) for n in nums]
        if len(nums) >= 2:
            return [nums[0], nums[1]]
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

def write_final_report(log_file: str) -> None:
    """
    Calculates and appends the final global evaluation metrics to the log file.
    Reads directly from the log file to account for current and historical runs.
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

    report = (
        f"\n================ VideoLLaMA3 Global Eval Report (Full Video + Context) ================\n"
        f"Total Evaluated Samples (incl. checkpoints) : {total}\n"
        f"Missing Files (MISSING)                     : {missing_count}\n"
        f"Exceptions/Errors (ERROR)                   : {error_count}\n"
        f"mIoU                                        : {(np.mean(all_ious) * 100):.2f}%\n"
        f"R@1 (IoU >= 0.3)                            : {(r1_03 / total * 100):.2f}%\n"
        f"R@1 (IoU >= 0.5)                            : {(r1_05 / total * 100):.2f}%\n"
        f"R@1 (IoU >= 0.7)                            : {(r1_07 / total * 100):.2f}%\n"
        f"========================================================================================\n"
    )
    
    print(report)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(report)

# ================= Main Execution Flow =================

def main():
    # 0. Create log directory
    if OUTPUT_LOG_FILE:
        log_dir = os.path.dirname(OUTPUT_LOG_FILE)
        if log_dir and not os.path.exists(log_dir): 
            os.makedirs(log_dir, exist_ok=True)

    # 1. Load Model
    print(f"📦 Loading Model: {MODEL_PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
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

    print(f"🚀 Start Inference (Complete Video Analysis + BG) -> {OUTPUT_LOG_FILE}")
    with open(OUTPUT_LOG_FILE, write_mode, encoding='utf-8') as f_log:
        if need_header:
            # Standardized 7-column header tracking PromptMsg for context validation
            f_log.write("ID\tPhase\tPred_Abs\tGT_Abs\tIoU\tRawResponse\tPromptMsg\n")
            f_log.flush()
        
        for item in tqdm(data, desc="Evaluating Full Video"):
            item_id = item.get('id', 'unknown')
            phase_name = item.get('phase', 'unknown phase')
            
            # Checkpoint skipping
            if item_id in processed_ids:
                continue

            # --- Video Path Resolution ---
            video_path = resolve_video_path(item_id)
            if not video_path or not os.path.exists(video_path):
                ious.append(0.0)
                f_log.write(f"{item_id}\t{phase_name}\t[0,0]\t[]\t0.0\tMISSING_FILE\tMISSING\n")
                f_log.flush()
                continue

            # --- Extract Ground Truth (Absolute time for full video prediction) ---
            gt_segments = []
            if 'time' in item:
                for t in item['time']:
                    if 's0' in t and 'e0' in t:
                        gt_segments.append([float(t['s0']), float(t['e0'])])
                    elif 's1' in t and 'e1' in t:
                        gt_segments.append([float(t['s1']), float(t['e1'])])
                        
            if not gt_segments:
                f_log.write(f"{item_id}\t{phase_name}\t[0,0]\t[]\t0.0\tNO_GT\tNO_GT\n")
                f_log.flush()
                continue

            try:
                # --- Core Modification: Extract 'query1' directly from JSON ---
                query_text = ""
                if 'conversation' in item and len(item['conversation']) > 0:
                    query_text = item['conversation'][0].get('query1', '')
                
                # Fallback if 'query1' is absent
                if not query_text:
                    query_text = f"For the surgery, could you specify the timestamps of {phase_name}?"
                
                # --- Extract Background Knowledge ---
                conv_data = item.get('conversation', [{}])[0]
                objective = conv_data.get('objective', '')
                anatomy = conv_data.get('key anatomical structures', '')
                steps = conv_data.get('key steps', '')

                bg_text = ""
                if objective or anatomy or steps:
                    bg_text += "Background Knowledge:\n"
                    if objective: bg_text += f"- Objective: {objective}\n"
                    if anatomy: bg_text += f"- Anatomy: {anatomy}\n"
                    if steps: bg_text += f"- Key Steps: {steps}\n\n"

                # --- Construct Final Prompt with BG and Strict Formatting ---
                final_prompt = (
                    f"You are a surgical assistant analyzing a full surgical video.\n\n"
                    f"{bg_text}"
                    f"Question: {query_text}\n"
                    f"Format your answer strictly as: [start, end]"
                )

                # --- Assemble Input Parameters (Ensuring max_frames aligns with baseline) ---
                conversation = [
                    {"role": "system", "content": "You are a helpful surgical assistant."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video", 
                                "video": {"video_path": video_path, "max_frames": MAX_FRAMES}
                            },
                            {"type": "text", "text": final_prompt},
                        ]
                    },
                ]

                inputs = processor(conversation=conversation, return_tensors="pt")
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                # --- Model Inference ---
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=128)
                
                response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                
                # Sanitize response and prompt to prevent log structure disruption
                if "assistant" in response:
                    response = response.split("assistant")[-1]
                clean_resp = response.replace('\n', ' ').replace('\t', ' ').strip()
                clean_msg = final_prompt.replace('\n', ' ').replace('\t', ' ').strip()
                
                # --- Parse Predictions & Calculate Results ---
                pred_timestamp = parse_model_response(clean_resp)
                max_iou = calculate_max_iou(pred_timestamp, gt_segments)
                ious.append(max_iou)

                # Write to standard 7-column log
                f_log.write(f"{item_id}\t{phase_name}\t{pred_timestamp}\t{gt_segments}\t{max_iou:.4f}\t{clean_resp}\t{clean_msg}\n")
                f_log.flush()

            except Exception as e:
                f_log.write(f"{item_id}\t{phase_name}\t[0,0]\t[]\t0.0\tERROR: {str(e)}\tERROR\n")
                f_log.flush()
                ious.append(0.0)

    # ================= Generate Final Evaluation Report =================
    write_final_report(OUTPUT_LOG_FILE)

if __name__ == "__main__":
    main()

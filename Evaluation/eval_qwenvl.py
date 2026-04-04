import os
import json
import torch
import numpy as np
import re
from typing import List, Tuple, Optional, Set, Any
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import warnings

warnings.filterwarnings("ignore")

# ================= Configuration =================
MODEL_PATH = ""
DATASET_JSON_PATH = "" 
OUTPUT_LOG_FILE = ""

# Dataset root directory mapping
DATA_ROOTS = {
    'pitvis':     '',
    'm2cai16':    '',
    'mvp57':      '',
    'autolaparo': '',
    'cataract':   ''
}

# Number of frames to sample per video
MAX_NUM_FRAMES = 100

# ================= Utility Functions =================

def resolve_video_path(item_id: str) -> Optional[str]:
    """
    Resolves the local video path based on the dataset prefix and item ID.
    """
    try:
        parts = str(item_id).lower().split('_')
        prefix = parts[0]
        vid_str = parts[1]
        folder = DATA_ROOTS.get(prefix)
        
        if not folder: 
            return None
            
        vid_num = int(vid_str)
        for ext in ['.mp4', '.avi']:
            path = os.path.join(folder, f"{vid_num}{ext}")
            if os.path.exists(path): 
                return path
        return os.path.join(folder, f"{vid_num}.mp4")
    except Exception:
        return None

def calculate_iou(pred: List[float], gt: List[float]) -> float:
    """
    Calculates the Temporal Intersection over Union (IoU) between a prediction and ground truth.
    """
    start_pred, end_pred = pred
    start_gt, end_gt = gt
    
    if start_pred > end_pred: 
        start_pred, end_pred = end_pred, start_pred
        
    intersection = max(0, min(end_pred, end_gt) - max(start_pred, start_gt))
    union = (end_pred - start_pred) + (end_gt - start_gt) - intersection
    
    return 0.0 if union == 0 else intersection / union

def calculate_max_iou(pred: List[float], gt_list: List[List[float]]) -> float:
    """
    Calculates the maximum IoU against a list of possible ground truth segments.
    """
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
        # Priority match: brackets containing two numbers [start, end]
        match = re.search(r'\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]', response)
        if match:
            return [float(match.group(1)), float(match.group(2))]
        
        # Fallback match: any two consecutive numbers
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        if len(nums) >= 2:
            return [float(nums[0]), float(nums[1])]
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
    Calculates and appends the final evaluation metrics to the log file.
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
        return

    r1_03 = sum(1 for x in all_ious if x >= 0.3)
    r1_05 = sum(1 for x in all_ious if x >= 0.5)
    r1_07 = sum(1 for x in all_ious if x >= 0.7)

    report = (
        f"\n================ Qwen2.5-VL Global Evaluation Report (Full Video + Context) ================\n"
        f"Total Evaluated Samples (incl. checkpoints) : {total}\n"
        f"Missing Files (MISSING)                     : {missing_count}\n"
        f"Exceptions/Errors (ERROR)                   : {error_count}\n"
        f"mIoU                                        : {(np.mean(all_ious) * 100):.2f}%\n"
        f"R@1 (IoU >= 0.3)                            : {(r1_03 / total * 100):.2f}%\n"
        f"R@1 (IoU >= 0.5)                            : {(r1_05 / total * 100):.2f}%\n"
        f"R@1 (IoU >= 0.7)                            : {(r1_07 / total * 100):.2f}%\n"
        f"====================================================================================\n"
    )
    
    print(report)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(report)

# ================= Main Execution Flow =================

def main():
    log_dir = os.path.dirname(OUTPUT_LOG_FILE)
    if log_dir and not os.path.exists(log_dir): 
        os.makedirs(log_dir, exist_ok=True)

    print(f"📦 Loading Model: {MODEL_PATH}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print(f"📖 Loading Dataset: {DATASET_JSON_PATH}")
    with open(DATASET_JSON_PATH, 'r') as f:
        data = json.load(f)
        
    if isinstance(data, dict):
        if 'dataset' in data: 
            data = data['dataset']
        elif 'data' in data: 
            data = data['data']

    # Scan for existing checkpoints
    processed_ids = get_processed_ids(OUTPUT_LOG_FILE)
    print(f"📌 Detected processed samples: {len(processed_ids)}")
    
    write_mode = 'a' if len(processed_ids) > 0 else 'w'
    need_header = len(processed_ids) == 0

    ious = []

    print(f"🚀 Start Inference (Complete Video + Background Context) -> {OUTPUT_LOG_FILE}")
    
    with open(OUTPUT_LOG_FILE, write_mode, encoding='utf-8') as f_log:
        if need_header:
            # Standard 7-column header
            f_log.write("ID\tPhase\tPred_Abs\tGT_Abs\tIoU\tRawResponse\tPromptMsg\n")
            f_log.flush()

        for item in tqdm(data, desc="Evaluating Full Video"):
            item_id = item.get('id', 'unknown')
            phase_name = item.get('phase', 'unknown phase')
            
            # Skip if already processed (checkpointing)
            if item_id in processed_ids:
                continue
            
            # 1. Resolve video path
            video_path = resolve_video_path(item_id)
            if not video_path or not os.path.exists(video_path):
                ious.append(0.0)
                f_log.write(f"{item_id}\t{phase_name}\t[0,0]\t[]\t0.0\tMISSING_FILE\tMISSING\n")
                f_log.flush()
                continue

            # 2. Extract Ground Truth (Absolute time for full video prediction)
            gt_segments_abs = []
            if 'time' in item:
                for t in item['time']:
                    gt_segments_abs.append([float(t['s0']), float(t['e0'])])
                    
            if not gt_segments_abs:
                f_log.write(f"{item_id}\t{phase_name}\t[0,0]\t[]\t0.0\tNO_GT\tNO_GT\n")
                f_log.flush()
                continue

            try:
                # 3. Extract and construct BG Prompt with background context
                conv_data = item.get('conversation', [{}])[0]
                
                # --- NEW: Extracting query1 directly from JSON ---
                json_query = conv_data.get('query1', f"Could you specify the timestamps of {phase_name}?")
                
                objective = conv_data.get('objective', '')
                anatomy = conv_data.get('key anatomical structures', '')
                steps = conv_data.get('key steps', '')

                bg_text = ""
                if objective or anatomy or steps:
                    bg_text += "Background Knowledge:\n"
                    if objective: bg_text += f"- Objective: {objective}\n"
                    if anatomy: bg_text += f"- Anatomy: {anatomy}\n"
                    if steps: bg_text += f"- Key Steps: {steps}\n\n"

                # Instruct the model to analyze the full video, providing the JSON query
                final_prompt = (
                    f"You are a surgical assistant. Analyze the surgical video provided.\n"
                    f"{bg_text}"
                    f"Question: {json_query}\n"
                    f"Format your answer strictly as: [start, end]"
                )

                # 4. Construct Qwen2.5-VL specific messages
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "video", 
                            "video": video_path,
                            "nframes": MAX_NUM_FRAMES
                        },
                        {"type": "text", "text": final_prompt}
                    ]
                }]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(model.device)

                # 5. Model Inference
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False  # Deterministic decoding
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()

                # 6. Post-processing
                clean_resp = response.replace('\n', ' ').replace('\t', ' ').strip()
                clean_msg = final_prompt.replace('\n', ' ').replace('\t', ' ').strip()
                
                # Parse the response for absolute timestamps
                pred_abs = parse_model_response(response)

                # 7. Calculate Max IoU
                max_iou = calculate_max_iou(pred_abs, gt_segments_abs)
                ious.append(max_iou)

                # Append results to log
                f_log.write(f"{item_id}\t{phase_name}\t{pred_abs}\t{gt_segments_abs}\t{max_iou:.4f}\t{clean_resp}\t{clean_msg}\n")
                f_log.flush()

            except Exception as e:
                f_log.write(f"{item_id}\t{phase_name}\t[0,0]\t[]\t0.0\tERROR: {str(e)}\tERROR\n")
                f_log.flush()
                ious.append(0.0)

    # 8. Generate and save the final report
    write_final_report(OUTPUT_LOG_FILE)

if __name__ == "__main__":
    main()
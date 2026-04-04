import os
import argparse
import json
import numpy as np
import torch
import re
import warnings
from typing import List, Tuple, Optional, Set, Any
from tqdm import tqdm

# ================= Environment Configuration =================
# 1. Clear proxy settings to prevent connection issues
proxies = ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']
for p in proxies:
    if p in os.environ:
        del os.environ[p]

# 2. Set HuggingFace mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 3. Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# ================= TimeChat Imports =================
from timechat.common.config import Config
from timechat.common.registry import registry
from timechat.conversation.conversation_video import Chat, conv_llava_llama_2

# ================= Configuration =================
# TimeChat configuration path
CFG_PATH = "" 
# Model checkpoint path
CKPT_PATH = "" 

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
    'cataract':   ''
}

# Inference parameters
GPU_ID = 0
NUM_FRAMES = 96  # Recommended frame count for TimeChat
NUM_BEAMS = 1
TEMPERATURE = 0.2

# ================= Utility Functions =================

class Args:
    """Mock argparse parameters for TimeChat config initialization."""
    def __init__(self):
        self.cfg_path = CFG_PATH
        self.gpu_id = GPU_ID
        self.options = None

def resolve_video_path(item_id: str) -> Optional[str]:
    """
    Resolves the absolute video path based on the item ID.
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
            
        return p1 # Default to mp4 if not found
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
    
    if union == 0: 
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
    Parses the TimeChat text response to extract start and end timestamps.
    """
    try:
        response = response.strip()
        # 1. Priority match: Brackets containing timestamps
        if '[' in response and ']' in response:
            content = response[response.find('[')+1 : response.find(']')]
            parts = content.split(',')
            if len(parts) >= 2:
                s = float(re.sub(r"[^0-9.]", "", parts[0]))
                e = float(re.sub(r"[^0-9.]", "", parts[1]))
                return [s, e]
        
        # 2. Fallback match: Extract any two consecutive numbers
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        nums = [float(n) for n in nums]
        if len(nums) >= 2:
            return [nums[0], nums[1]]
    except Exception:
        pass
    return [0.0, 0.0]

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
        f"\n================ TimeChat Global Evaluation Report (Full Video + Context) ================\n"
        f"Total Evaluated Samples (incl. checkpoints) : {total}\n"
        f"Missing Files (MISSING)                     : {missing_count}\n"
        f"Exceptions/Errors (ERROR)                   : {error_count}\n"
        f"mIoU                                        : {(np.mean(all_ious) * 100):.2f}%\n"
        f"R@1 (IoU >= 0.3)                            : {(r1_03 / total * 100):.2f}%\n"
        f"R@1 (IoU >= 0.5)                            : {(r1_05 / total * 100):.2f}%\n"
        f"R@1 (IoU >= 0.7)                            : {(r1_07 / total * 100):.2f}%\n"
        f"========================================================================================\n"
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
    # 1. Initialize Model
    print("📦 Initializing TimeChat...")
    args = Args()
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_config.ckpt = CKPT_PATH
    
    # Load Model
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model.eval()

    # Load Processor
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    # Initialize Chat interface
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('✅ Initialization Finished')

    # 2. Load Dataset
    print(f"📖 Loading Dataset: {DATASET_JSON_PATH}")
    with open(DATASET_JSON_PATH, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get('dataset', data.get('data', []))
    print(f"Total samples to evaluate: {len(data)}")

    # 3. Setup Logging & Checkpoint Resumption
    if OUTPUT_LOG_FILE:
        os.makedirs(os.path.dirname(OUTPUT_LOG_FILE), exist_ok=True)
    
    processed_ids = get_processed_ids(OUTPUT_LOG_FILE)
    print(f"📌 Detected processed samples: {len(processed_ids)}")
    
    write_mode = 'a' if len(processed_ids) > 0 else 'w'
    need_header = len(processed_ids) == 0

    ious = []
    missing_files = []

    print(f"🚀 Start Inference (Complete Video + Background Context) -> {OUTPUT_LOG_FILE}")
    with open(OUTPUT_LOG_FILE, write_mode, encoding='utf-8') as f_log:
        if need_header:
            f_log.write("ID\tPhase\tPred_Abs\tGT_Abs\tIoU\tRawResponse\tPromptMsg\n")
            f_log.flush()
        
        for item in tqdm(data, desc="Evaluating Full Video"):
            item_id = item.get('id', 'unknown')
            
            # Checkpoint skipping
            if item_id in processed_ids:
                continue
                
            phase = item.get('phase', '')
            surgery_name = item.get('surgery', 'surgery')
            
            # Resolve paths
            video_path = resolve_video_path(item_id)
            
            # File validation
            if not video_path or not os.path.exists(video_path):
                missing_files.append(f"{item_id} -> {video_path}")
                ious.append(0.0)
                f_log.write(f"{item_id}\t{phase}\t[0,0]\t[]\t0.0\tMISSING_FILE\tMISSING\n")
                f_log.flush()
                continue

            # Extract Ground Truth
            gt_segments = []
            if 'time' in item:
                for t in item['time']:
                    if 's0' in t and 'e0' in t:
                        gt_segments.append([float(t['s0']), float(t['e0'])])

            try:
                # A. Reset chat state and set System Prompt
                img_list = []
                chat_state = conv_llava_llama_2.copy()
                chat_state.system = f"You are given a video from the {surgery_name} dataset."

                # B. Upload full video
                msg = chat.upload_video_without_audio(
                    video_path=video_path, 
                    conv=chat_state,
                    img_list=img_list, 
                    n_frms=NUM_FRAMES
                )

                # C. Extract and construct Background Knowledge Prompt
                conv_data = item.get('conversation', [{}])[0]
                json_query = conv_data.get('query1', f"Could you specify the timestamps of {phase}?")
                
                objective = conv_data.get('objective', '')
                anatomy = conv_data.get('key anatomical structures', '')
                steps = conv_data.get('key steps', '')

                bg_text = ""
                if objective or anatomy or steps:
                    bg_text += "Background Knowledge:\n"
                    if objective: bg_text += f"- Objective: {objective}\n"
                    if anatomy: bg_text += f"- Anatomy: {anatomy}\n"
                    if steps: bg_text += f"- Key Steps: {steps}\n\n"

                # Incorporate context and strict formatting instructions
                text_input = (
                    f"Please find the visual event described by a sentence in the video, "
                    f"determining its starting and ending times in seconds.\n"
                    f"{bg_text}"
                    f"The format should strictly be: 'The event happens in the [start] - [end] seconds'. "
                    f"For example, The event 'person turn a light on' happens in the [24.3] - [30.4] seconds. "
                    f"Now I will give you the textual sentence: {json_query}. Please return its start time and end time."
                )
                
                # D. Model Inference
                chat.ask(text_input, chat_state)
                llm_message = chat.answer(
                    conv=chat_state,
                    img_list=img_list,
                    num_beams=NUM_BEAMS,
                    temperature=TEMPERATURE,
                    max_new_tokens=64,
                    max_length=2000
                )[0]

                # E. Result Parsing & Logging
                clean_resp = llm_message.replace('\n', ' ').replace('\t', ' ').strip()
                actual_prompt = chat_state.messages[-2][1] 
                clean_msg = actual_prompt.replace('\n', ' ').replace('\t', ' ').strip()
                
                pred_timestamp = parse_model_response(llm_message)
                max_iou = calculate_max_iou(pred_timestamp, gt_segments)
                ious.append(max_iou)

                # Write to log
                f_log.write(f"{item_id}\t{phase}\t{pred_timestamp}\t{gt_segments}\t{max_iou:.4f}\t{clean_resp}\t{clean_msg}\n")
                f_log.flush()

            except Exception as e:
                f_log.write(f"{item_id}\t{phase}\t[0,0]\t[]\t0.0\tERROR: {str(e)}\tERROR\n")
                f_log.flush()
                ious.append(0.0)

    # 4. Generate Final Evaluation Report
    write_final_report(OUTPUT_LOG_FILE, missing_files)

if __name__ == "__main__":
    main()
import os
import sys
import torch
import json
import numpy as np
import re
from typing import List, Tuple, Optional, Set, Dict, Any
from tqdm import tqdm

# ================= VTimeLLM Path Configuration =================
ROOT_DIR = "" 
if ROOT_DIR:
    sys.path.append(ROOT_DIR)

try:
    from vtimellm.constants import IMAGE_TOKEN_INDEX
    from vtimellm.conversation import conv_templates, SeparatorStyle
    from vtimellm.model.builder import load_pretrained_model
    from vtimellm.utils import disable_torch_init
    from vtimellm.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, VideoExtractor
    import clip
    from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"Please ensure ROOT_DIR ({ROOT_DIR}) is correct and all dependencies are installed.")
    sys.exit(1)

# ================= Configuration =================
# Model weight configurations
MODEL_BASE = ""
STAGE1_PATH = ""
STAGE2_PATH = ""
STAGE3_PATH = ""
CLIP_PATH = ""

# Dataset JSON path containing background knowledge
DATASET_JSON_PATH = ""
# Output log file path (e.g., phase_complete_bg-2.txt to compare against baseline)
OUTPUT_LOG_FILE = ""

# Dataset root directory mapping
DATA_ROOTS = {
    'pitvis':     '',
    'm2cai16':    '',
    'mvp57':      '',
    'autolaparo': '',
    'cataract':   ''
}

# Number of frames to extract (strictly aligned with baseline)
NUM_EXTRACT_FRAMES = 100

# ================= Inference Function =================

def inference(model: Any, image: torch.Tensor, query: str, tokenizer: Any) -> str:
    """
    Generates a response from the VTimeLLM model given video features and a text query.
    """
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image[None,].cuda(), 
            do_sample=True,
            temperature=0.05,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True
        )

    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
        
    return outputs.strip()

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
    """Calculates the Temporal Intersection over Union (IoU)."""
    start_pred, end_pred = pred
    start_gt, end_gt = gt
    
    if start_pred > end_pred: 
        start_pred, end_pred = end_pred, start_pred
        
    intersection = max(0, min(end_pred, end_gt) - max(start_pred, start_gt))
    union = (end_pred - start_pred) + (end_gt - start_gt) - intersection
    
    return 0.0 if union == 0 else intersection / union

def calculate_max_iou(pred: List[float], gt_list: List[List[float]]) -> float:
    """Calculates the maximum IoU against a list of ground truth segments."""
    if not gt_list: 
        return 0.0
    ious = [calculate_iou(pred, gt) for gt in gt_list]
    return max(ious)

def parse_vtimellm_response(response: str, full_video_duration: float) -> List[float]:
    """
    Parses VTimeLLM's standard output format 'xx to yy' (relative percentage 0-100).
    For full video evaluation, multiplies the percentage by the full video duration
    to yield absolute timestamps in seconds.
    """
    try:
        matches = re.search(r"(\d{1,3})\s*(?:to|and)\s*(\d{1,3})", response)
        if matches:
            from_percent = float(matches.group(1)) / 100.0
            to_percent = float(matches.group(2)) / 100.0
            
            # Clamp values between 0.0 and 1.0
            from_percent = max(0.0, min(1.0, from_percent))
            to_percent = max(0.0, min(1.0, to_percent))
            
            if from_percent > to_percent:
                from_percent, to_percent = to_percent, from_percent
            
            s = from_percent * full_video_duration
            e = to_percent * full_video_duration
            return [s, e]
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
        f"\n================ VTimeLLM Global Eval Report (Full Video + Context) ================\n"
        f"Total Evaluated Samples (incl. checkpoints) : {total}\n"
        f"Missing Files (MISSING)                     : {missing_count}\n"
        f"Exceptions/Errors (ERROR)                   : {error_count}\n"
        f"mIoU                                        : {(np.mean(all_ious) * 100):.2f}%\n"
        f"R@1 (IoU >= 0.3)                            : {(r1_03 / total * 100):.2f}%\n"
        f"R@1 (IoU >= 0.5)                            : {(r1_05 / total * 100):.2f}%\n"
        f"R@1 (IoU >= 0.7)                            : {(r1_07 / total * 100):.2f}%\n"
        f"======================================================================================\n"
    )
    
    print(report)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(report)

# ================= Main Execution Flow =================

def main():
    class Args:
        model_base = MODEL_BASE
        pretrain_mm_mlp_adapter = STAGE1_PATH
        stage2 = STAGE2_PATH
        stage3 = STAGE3_PATH
        clip_path = CLIP_PATH

    args = Args()

    # 1. Initialize Models
    print("📦 Initializing Models...")
    disable_torch_init()
    
    # Load VTimeLLM components
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3)
    model = model.cuda().to(torch.float16)
    
    # Always load CLIP and VideoExtractor for dynamic feature extraction on full videos
    print(f"🎥 Dynamically extracting features using CLIP from {args.clip_path}...")
    clip_model, _ = clip.load(args.clip_path)
    clip_model.eval().cuda()
    
    video_loader = VideoExtractor(N=NUM_EXTRACT_FRAMES)
    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # 2. Load Dataset
    print(f"📖 Loading Dataset: {DATASET_JSON_PATH}")
    with open(DATASET_JSON_PATH, 'r') as f:
        data = json.load(f)
        
    if isinstance(data, dict):
        if 'dataset' in data: 
            data = data['dataset']
        elif 'data' in data: 
            data = data['data']

    # 3. Setup Logging & Checkpoint Resumption
    if OUTPUT_LOG_FILE:
        os.makedirs(os.path.dirname(OUTPUT_LOG_FILE), exist_ok=True)
        
    processed_ids = get_processed_ids(OUTPUT_LOG_FILE)
    print(f"📌 Detected processed samples: {len(processed_ids)}")
    
    write_mode = 'a' if len(processed_ids) > 0 else 'w'
    need_header = len(processed_ids) == 0

    print(f"🚀 Start Inference (Complete Video + Background Context) -> {OUTPUT_LOG_FILE}")
    with open(OUTPUT_LOG_FILE, write_mode, encoding='utf-8') as f_log:
        if need_header:
            # 7-column header tracking PromptMsg for context validation
            f_log.write("ID\tQuery\tPred_Abs\tGT_Abs\tIoU\tRawResponse\tPromptMsg\n")

        for item in tqdm(data, desc="Evaluating Complete Video"):
            item_id = item.get('id', 'unknown')
            
            # Checkpoint skipping
            if item_id in processed_ids:
                continue
            
            # Fetch full video duration
            full_video_duration = float(item.get('duration', 0.0))
            if full_video_duration == 0.0:
                full_video_duration = 3600.0  # Safe fallback
            
            # --- Extract Background Knowledge & Query ---
            conv_data = item.get('conversation', [{}])[0]
            
            # Look for query12, fallback to query1, then generic phase query
            query_text = conv_data.get('query12', conv_data.get('query1', ''))
            phase = item.get('phase', '')
            if not query_text:
                query_text = f"Please identify the time period when '{phase}' occurs in seconds."

            objective = conv_data.get('objective', '')
            anatomy = conv_data.get('key anatomical structures', '')
            steps = conv_data.get('key steps', '')

            bg_text = ""
            if objective or anatomy or steps:
                bg_text += "Background Knowledge:\n"
                if objective: bg_text += f"- Objective: {objective}\n"
                if anatomy: bg_text += f"- Anatomy: {anatomy}\n"
                if steps: bg_text += f"- Key Steps: {steps}\n\n"

            # Sanitized query string for safe log column entry
            log_query_name = query_text.replace('\n', ' ').replace('\t', ' ')

            # --- Extract Ground Truth (Absolute time for full video prediction) ---
            gt_ts_list = []
            if 'time' in item:
                for t in item['time']:
                    if 's0' in t and 'e0' in t:
                        gt_ts_list.append([float(t['s0']), float(t['e0'])])
                    elif 's1' in t and 'e1' in t:
                        gt_ts_list.append([float(t['s1']), float(t['e1'])])
            
            if not gt_ts_list:
                f_log.write(f"{item_id}\t{log_query_name}\t[0,0]\t[]\t0.0\tNO_GT\tNO_GT\n")
                f_log.flush()
                continue

            # --- Resolve Video Path ---
            video_path = resolve_video_path(item_id)
            if not video_path or not os.path.exists(video_path):
                f_log.write(f"{item_id}\t{log_query_name}\t[0,0]\t[]\t0.0\tMISSING\tMISSING\n")
                f_log.flush()
                continue
                
            try:
                # ================= Dynamic Full Video Feature Extraction =================
                # 1. Uniformly extract N=100 frames (Strict baseline alignment)
                _, images = video_loader.extract({'id': None, 'video': video_path})
                # 2. Preprocess images
                images = transform(images / 255.0)
                images = images.to(torch.float16)
                # 3. CLIP Encoding
                with torch.no_grad():
                    features = clip_model.encode_image(images.to('cuda'))
                # =========================================================================
                
                # --- Construct Final Prompt with Context ---
                # VTimeLLM Expected Format: <video>\n + bg_text + query
                full_query = f"<video>\n{bg_text}Question: {query_text}"

                # --- Model Inference ---
                response = inference(model, features, full_query, tokenizer)
                clean_resp = response.strip()

                # --- Post-processing ---
                pred_abs = parse_vtimellm_response(clean_resp, full_video_duration)
                max_iou = calculate_max_iou(pred_abs, gt_ts_list)

                # --- Logging ---
                clean_resp_log = clean_resp.replace('\n', ' ').replace('\t', ' ')
                clean_full_query_log = full_query.replace('\n', ' ').replace('\t', ' ')
                
                f_log.write(f"{item_id}\t{log_query_name}\t{pred_abs}\t{gt_ts_list}\t{max_iou:.4f}\t{clean_resp_log}\t{clean_full_query_log}\n")
                f_log.flush()

            except Exception as e:
                f_log.write(f"{item_id}\t{log_query_name}\t[0,0]\t[]\t0.0\tERROR: {str(e)}\tERROR\n")
                f_log.flush()

    # --- Script Completion: Calculate final metrics and append to log ---
    write_final_report(OUTPUT_LOG_FILE)

if __name__ == "__main__":
    main()

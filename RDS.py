import os
import numpy as np
import pandas as pd
import decord
from typing import Tuple, Optional

# ================= Configuration =================
# Sampling and redundancy filtering parameters
TARGET_FRAMES = 100        # Target number of frames to sample
PARAM_P_PERCENTILE = 40    # P40 percentile threshold for similarity
PARAM_W_WINDOW = 5         # Rolling window size for variance calculation
PARAM_V_VARIANCE = 0.005   # Variance threshold for determining stability
PARAM_MIN_DUR = 12         # Minimum duration (in frames) to be considered a stable segment
# ==========================================

# --- Core Logic: Load Full Video ---
def load_full_video(video_path: str) -> Tuple[Optional[np.ndarray], float]:
    """
    Loads a full video using Decord, decodes frames at a lower resolution 
    to save memory, and returns the frame array and FPS.
    """
    if not os.path.exists(video_path):
        return None, 0.0

    try:
        # Read video with Decord, specifying a smaller resolution for memory optimization
        vr = decord.VideoReader(video_path, width=336, height=336, ctx=decord.cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        
        if total_frames <= 0:
            return None, fps
            
        indices = list(range(total_frames))
        frames = vr.get_batch(indices).asnumpy()
        return frames, fps
        
    except Exception as e:
        print(f"[ERROR] Failed to read video at {video_path}: {e}")
        return None, 0.0

# --- Dynamic Sampling Logic ---
def extract_rgb_mean_features(frames_np: np.ndarray) -> np.ndarray:
    """
    Downsamples the spatial dimensions of the frames and extracts 
    the mean RGB features for similarity calculation.
    """
    small_frames = frames_np[:, ::4, ::4, :] 
    return np.mean(small_frames, axis=(1, 2))

def calculate_redundancy_mask(sims: np.ndarray, fps: float) -> np.ndarray:
    """
    Calculates a boolean mask identifying redundant frame transitions 
    based on similarity percentiles and local variance stability.
    """
    if len(sims) == 0: 
        return np.zeros(0, dtype=bool)
    
    df = pd.DataFrame({'sim': sims})
    threshold_p = np.percentile(sims, PARAM_P_PERCENTILE)
    
    # Identify high similarity and low variance (stable) segments
    mask_high_sim = df['sim'] >= threshold_p
    rolling_var = df['sim'].rolling(window=PARAM_W_WINDOW, center=True).var().bfill().ffill()
    mask_stable = rolling_var < PARAM_V_VARIANCE
    candidate_mask = mask_high_sim & mask_stable
    
    final_mask = np.zeros_like(candidate_mask, dtype=bool)
    candidates = candidate_mask.values.astype(int)
    
    # Find start and end indices of stable segments
    changes = np.diff(np.concatenate(([0], candidates, [0])))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    
    # Adjust minimum duration threshold based on video FPS
    min_dur_frames = PARAM_MIN_DUR
    if fps < 5.0: 
        min_dur_frames = 2 
    
    # Apply mask only to segments exceeding the minimum duration
    for s, e in zip(starts, ends):
        if (e - s) >= min_dur_frames:
            final_mask[s:e] = True
            
    return final_mask

def get_weighted_indices(total_frames: int, static_mask: np.ndarray, target_count: int) -> np.ndarray:
    """
    Generates target frame indices by down-weighting redundant segments 
    and sampling uniformly across the resulting Cumulative Distribution Function (CDF).
    """
    weights = np.ones(total_frames)
    if len(static_mask) > 0:
        limit = min(len(static_mask), total_frames)
        mask_part = static_mask[:limit]
        weights[:limit][mask_part] = 0.05  # Down-weight redundant frames
    
    # Calculate CDF
    probs = weights / np.sum(weights)
    cdf = np.insert(np.cumsum(probs), 0, 0)
    cdf = cdf / cdf[-1]
    
    # Interpolate target indices
    target_y = np.linspace(0, 1, target_count)
    xp = np.linspace(0, total_frames - 1, len(cdf))
    indices = np.interp(target_y, cdf, xp)
    
    return np.round(indices).astype(int)

# --- Main Execution Function ---
def extract_and_save_indices(video_path: str, output_npy_path: str) -> None:
    """
    Executes the full pipeline: loads video, calculates dynamic redundancy, 
    samples target frames, and saves the indices to a numpy array file.
    """
    print(f"[INFO] Processing video: {video_path}")
    
    # 1. Load full video
    frames_np, fps = load_full_video(video_path)
    
    if frames_np is None or len(frames_np) == 0:
        print("[ERROR] Video loading failed or video is empty.")
        return

    total_len = len(frames_np)
    print(f"[INFO] Video loaded successfully | Total Frames: {total_len} | FPS: {fps:.2f}")
    
    # 2. Dynamic Sampling Logic
    if total_len <= TARGET_FRAMES:
        # Fallback to linear sampling if video is shorter than target frames
        selected_indices = np.linspace(0, total_len - 1, TARGET_FRAMES).astype(int)
        selected_indices = np.clip(selected_indices, 0, total_len - 1)
    else:
        # Extract features and calculate similarities
        rgb_feats = extract_rgb_mean_features(frames_np)
        norms = np.linalg.norm(rgb_feats, axis=1, keepdims=True)
        rgb_norm = rgb_feats / (norms + 1e-6)
        sims = np.sum(rgb_norm[:-1] * rgb_norm[1:], axis=1)
        
        # Calculate dynamic redundancy and retrieve weighted indices
        static_mask = calculate_redundancy_mask(sims, fps) 
        selected_indices = get_weighted_indices(total_len, static_mask, TARGET_FRAMES)

    # 3. Typecast to int64 and save to disk
    selected_indices = selected_indices.astype(np.int64)
    
    output_dir = os.path.dirname(output_npy_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    np.save(output_npy_path, selected_indices)
    
    print(f"[SUCCESS] Sampling complete. Total sampled frames: {len(selected_indices)}.")
    print(f"[SUCCESS] Indices saved to: {output_npy_path}")
    
    # 4. Print the final 100 frame indices
    print("\n[INFO] Complete list of sampled frame indices:")
    print(selected_indices.tolist())

if __name__ == "__main__":
    # ================= Usage Example =================
    # Define your input video path and desired output .npy path here
    INPUT_VIDEO_PATH = "" 
    OUTPUT_NPY_PATH = ""
    
    if INPUT_VIDEO_PATH and OUTPUT_NPY_PATH:
        extract_and_save_indices(INPUT_VIDEO_PATH, OUTPUT_NPY_PATH)
    else:
        print("[WARNING] Please specify INPUT_VIDEO_PATH and OUTPUT_NPY_PATH before running.")
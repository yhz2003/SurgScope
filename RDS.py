import os
import numpy as np
import pandas as pd
import decord

# ================= 配置区域 =================
# 采样与冗余过滤参数
TARGET_FRAMES = 100        # 目标采样帧数
PARAM_P_PERCENTILE = 40    # P40
PARAM_W_WINDOW = 5         # 窗口大小
PARAM_V_VARIANCE = 0.005   # 方差阈值
PARAM_MIN_DUR = 12         # 最小驻留帧数
# ==========================================

# --- 核心逻辑：加载完整视频 ---
def load_full_video(video_path):
    if not os.path.exists(video_path):
        return None, 0

    try:
        # 使用 decord 读取视频，指定较小的分辨率以节省内存
        vr = decord.VideoReader(video_path, width=336, height=336, ctx=decord.cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        
        if total_frames <= 0:
            return None, fps
            
        indices = list(range(total_frames))
        frames = vr.get_batch(indices).asnumpy()
        return frames, fps
        
    except Exception as e:
        print(f"⚠️ 读取视频出错 {video_path}: {e}")
        return None, 0

# --- 动态采样逻辑 ---
def extract_rgb_mean_features(frames_np):
    small_frames = frames_np[:, ::4, ::4, :] 
    return np.mean(small_frames, axis=(1, 2))

def calculate_redundancy_mask(sims, fps):
    if len(sims) == 0: return np.zeros(0, dtype=bool)
    
    df = pd.DataFrame({'sim': sims})
    threshold_p = np.percentile(sims, PARAM_P_PERCENTILE)
    mask_high_sim = df['sim'] >= threshold_p
    rolling_var = df['sim'].rolling(window=PARAM_W_WINDOW, center=True).var().bfill().ffill()
    mask_stable = rolling_var < PARAM_V_VARIANCE
    candidate_mask = mask_high_sim & mask_stable
    
    final_mask = np.zeros_like(candidate_mask, dtype=bool)
    candidates = candidate_mask.values.astype(int)
    changes = np.diff(np.concatenate(([0], candidates, [0])))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    
    min_dur_frames = PARAM_MIN_DUR
    if fps < 5.0: 
        min_dur_frames = 2 
    
    for s, e in zip(starts, ends):
        if (e - s) >= min_dur_frames:
            final_mask[s:e] = True
    return final_mask

def get_weighted_indices(total_frames, static_mask, target_count):
    weights = np.ones(total_frames)
    if len(static_mask) > 0:
        limit = min(len(static_mask), total_frames)
        mask_part = static_mask[:limit]
        weights[:limit][mask_part] = 0.05 
    
    probs = weights / np.sum(weights)
    cdf = np.insert(np.cumsum(probs), 0, 0)
    cdf = cdf / cdf[-1]
    target_y = np.linspace(0, 1, target_count)
    xp = np.linspace(0, total_frames - 1, len(cdf))
    indices = np.interp(target_y, cdf, xp)
    return np.round(indices).astype(int)

# --- 主处理函数 ---
def extract_and_save_indices(video_path, output_npy_path):
    print(f"⏳ 正在处理视频: {video_path}")
    
    # 1. 加载完整视频
    frames_np, fps = load_full_video(video_path)
    
    if frames_np is None or len(frames_np) == 0:
        print("❌ 视频加载失败或视频为空！")
        return

    total_len = len(frames_np)
    print(f"✅ 视频加载成功 | 总帧数: {total_len} | FPS: {fps:.2f}")
    
    # 2. 动态采样逻辑
    if total_len <= TARGET_FRAMES:
        selected_indices = np.linspace(0, total_len-1, TARGET_FRAMES).astype(int)
        selected_indices = np.clip(selected_indices, 0, total_len - 1)
    else:
        rgb_feats = extract_rgb_mean_features(frames_np)
        norms = np.linalg.norm(rgb_feats, axis=1, keepdims=True)
        rgb_norm = rgb_feats / (norms + 1e-6)
        sims = np.sum(rgb_norm[:-1] * rgb_norm[1:], axis=1)
        
        static_mask = calculate_redundancy_mask(sims, fps) 
        selected_indices = get_weighted_indices(total_len, static_mask, TARGET_FRAMES)

    # 3. 转换为 int64 并保存为 numpy 数组
    selected_indices = selected_indices.astype(np.int64)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_npy_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    np.save(output_npy_path, selected_indices)
    
    print(f"🎉 采样完成！共采样 {len(selected_indices)} 帧。")
    print(f"💾 索引已保存至: {output_npy_path}")
    
    # 4. 打印出完整的 100 帧索引号
    print("\n📋 完整的 100 帧索引号如下:")
    print(selected_indices.tolist())

if __name__ == "__main__":
    # ================= 使用示例 =================
    # 在这里指定你的输入视频路径和输出 .npy 路径
    INPUT_VIDEO_PATH = "your_video.mp4" 
    OUTPUT_NPY_PATH = "sampled_indices.npy"
    
    extract_and_save_indices(INPUT_VIDEO_PATH, OUTPUT_NPY_PATH)
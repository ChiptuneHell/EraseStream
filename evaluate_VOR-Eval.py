import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from tqdm import tqdm
import cv2
import os

def load_video(video_path):
    """
    Load video into numpy array with shape [T, H, W, C]
    """
    cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    return np.array(frames)

def resize_video(video, target_size):
    """
    Resize all frames in a video

    Args:
        video: numpy array [T,H,W,C]
        target_size: (width, height)

    Returns:
        resized video
    """
    resized_frames = []

    for frame in video:
        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        resized_frames.append(resized)

    return np.array(resized_frames)

class VideoQualityEvaluator:
    def __init__(self, device='cuda'):
        """Initialize video quality evaluator with specified computation device
        
        Args:
            device (str): Computation device ('cuda' or 'cpu')
        """
        self.device = device
        # Initialize LPIPS model (perceptual similarity metric)
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
    
    def _preprocess_frame(self, frame):
        """Convert frame to standardized format for evaluation
        
        Args:
            frame: Input frame (numpy array or torch tensor)
            
        Returns:
            Processed frame in HWC format with values in [0,1]
        """
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        
        # Normalize to [0,1] if needed
        if frame.max() > 1:
            frame = frame / 255.0
        # Convert CHW to HWC if needed
        if len(frame.shape) == 3 and frame.shape[0] == 3:
            frame = frame.transpose(1, 2, 0)
        return frame
    
    def calculate_psnr(self, vid1, vid2):
        """Calculate average PSNR between two videos
        
        Args:
            vid1: First video (list/array of frames)
            vid2: Second video (list/array of frames)
            
        Returns:
            Mean PSNR value across all frames
        """
        psnrs = []
        for f1, f2 in zip(vid1, vid2):
            f1 = self._preprocess_frame(f1)
            f2 = self._preprocess_frame(f2)
            # Calculate PSNR for this frame pair
            psnrs.append(psnr(f1, f2, data_range=1.0))
        return np.mean(psnrs)
    
    def calculate_ssim(self, vid1, vid2):
        """Calculate average SSIM between two videos
        
        Args:
            vid1: First video (list/array of frames)
            vid2: Second video (list/array of frames)
            
        Returns:
            Mean SSIM value across all frames
        """
        ssims = []
        for f1, f2 in zip(vid1, vid2):
            f1 = self._preprocess_frame(f1)
            f2 = self._preprocess_frame(f2)
            # Calculate SSIM for this frame pair (multichannel for color images)
            ssims.append(ssim(f1, f2, channel_axis=2, data_range=1.0))
        return np.mean(ssims)
    
    def calculate_lpips(self, vid1, vid2):
        """Calculate average LPIPS (perceptual similarity) between two videos
        
        Args:
            vid1: First video (list/array of frames)
            vid2: Second video (list/array of frames)
            
        Returns:
            Mean LPIPS value across all frames (lower is better)
        """
        lpips_values = []
        for f1, f2 in zip(vid1, vid2):
            # Convert to torch tensor if needed
            if not isinstance(f1, torch.Tensor):
                f1 = torch.from_numpy(f1).permute(2, 0, 1).unsqueeze(0).float()  # HWC -> 1CHW
                f2 = torch.from_numpy(f2).permute(2, 0, 1).unsqueeze(0).float()
            
            # Normalize to [-1,1] if needed
            if f1.max() > 1:
                f1 = f1 / 127.5 - 1.0
                f2 = f2 / 127.5 - 1.0
            
            f1 = f1.to(self.device)
            f2 = f2.to(self.device)
            
            # Calculate LPIPS with no gradients
            with torch.no_grad():
                lpips_values.append(self.lpips_model(f1, f2).item())
        return np.mean(lpips_values)
    
    def evaluate_videos(self, generated_video, reference_video, metrics=['psnr','lpips','ssim']):
        """Comprehensive video quality evaluation between generated and reference videos
        
        Args:
            generated_video: Model-generated video [T,H,W,C] or [T,C,H,W]
            reference_video: Ground truth reference video [T,H,W,C] or [T,C,H,W]
            metrics: List of metrics to compute ('psnr', 'ssim', 'lpips')
            
        Returns:
            Dictionary containing computed metric values
        """
        results = {}
        
        # Verify video lengths match
        assert len(generated_video) == len(reference_video), "Videos must have same number of frames"
        
        # Calculate requested metrics
        if 'psnr' in metrics:
            results['psnr'] = self.calculate_psnr(generated_video, reference_video)
        
        if 'ssim' in metrics:
            results['ssim'] = self.calculate_ssim(generated_video, reference_video)
        
        if 'lpips' in metrics:
            results['lpips'] = self.calculate_lpips(generated_video, reference_video)
        
        return results


def run_evaluation(gt_dir, pred_dir, device='cuda'):
    evaluator = VideoQualityEvaluator(device=device)
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

    subfolders = sorted([f for f in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, f))])

    subfolder_results = {}
    all_psnr, all_ssim, all_lpips = [], [], []

    print(f"找到 {len(subfolders)} 个子文件夹，开始高效评测...\n")
    
    # 打印表头
    header = f"{'子文件夹名称':<15} | {'处理视频数':<8} | {'PSNR (↑)':<10} | {'SSIM (↑)':<10} | {'LPIPS (↓)':<10}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for subfolder in subfolders:
        gt_sub_path = os.path.join(gt_dir, subfolder, "Edited")
        pred_sub_path = os.path.join(pred_dir, subfolder)

        if not os.path.exists(gt_sub_path) or not os.path.exists(pred_sub_path):
            continue

        gt_videos = sorted([f for f in os.listdir(gt_sub_path) if f.lower().endswith(video_extensions)])
        folder_psnr, folder_ssim, folder_lpips = [], [], []

        # leave=False 确保进度条在跑完当前文件夹后会自动消失，不破坏表格输出格式
        for video_name in tqdm(gt_videos, desc=f"评估 {subfolder}", leave=False):
            gt_video_file = os.path.join(gt_sub_path, video_name)
            pred_video_file = os.path.join(pred_sub_path, video_name)

            if not os.path.exists(pred_video_file):
                continue

            ref_video = load_video(gt_video_file)
            gen_video = load_video(pred_video_file)

            if len(ref_video) == 0 or len(gen_video) == 0:
                continue

            # =========================
            # Align frame count
            # =========================
            min_frames = min(len(gen_video), len(ref_video))
            gen_video = gen_video[:min_frames]
            ref_video = ref_video[:min_frames]

            # =========================
            # Align resolution
            # =========================
            gen_h, gen_w = gen_video[0].shape[:2]

            # resize reference video to generated video size
            ref_video = resize_video(ref_video, (gen_w, gen_h))

            # evaluate
            results = evaluator.evaluate_videos(
                gen_video,
                ref_video,
                metrics=['psnr', 'ssim', 'lpips']
            )

            folder_psnr.append(results['psnr'])
            folder_ssim.append(results['ssim'])
            folder_lpips.append(results['lpips'])

        # 当前子文件夹跑完后，立刻计算并【实时打印】
        if len(folder_psnr) > 0:
            avg_psnr = float(np.mean(folder_psnr))
            avg_ssim = float(np.mean(folder_ssim))
            avg_lpips = float(np.mean(folder_lpips))

            subfolder_results[subfolder] = {
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'lpips': avg_lpips,
                'count': len(folder_psnr)
            }

            all_psnr.extend(folder_psnr)
            all_ssim.extend(folder_ssim)
            all_lpips.extend(folder_lpips)

            # 实时打印一行结果
            print(f"{subfolder:<17} | {len(folder_psnr):<12} | {avg_psnr:<10.4f} | {avg_ssim:<10.4f} | {avg_lpips:<10.4f}")

    # ================= 打印总体汇总 =================
    print("-" * len(header))
    if len(all_psnr) > 0:
        total_psnr = float(np.mean(all_psnr))
        total_ssim = float(np.mean(all_ssim))
        total_lpips = float(np.mean(all_lpips))
        print(f"{'总体平均 (Total)':<17} | {len(all_psnr):<12} | {total_psnr:<10.4f} | {total_ssim:<10.4f} | {total_lpips:<10.4f}")
    else:
        print("未成功评估任何视频对！")
    print("=" * len(header))


if __name__ == "__main__":

    # 模型生成的视频的文件夹
    PRED_DIR = "/mnt/cpfs/epic-user/dongzhuobai-20260612/Erase-Forcing/0821/stage1/64000step"

    # 真实文件夹数据
    GT_DIR = "/mnt/cpfs/epic-user/dongzhuobai-20260612/Erase-Forcing/Benchmark/ROSE"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_evaluation(GT_DIR, PRED_DIR, device=device)
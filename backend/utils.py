# /backend/utils.py

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Callable
import pandas as pd
import torch
from transformers import AutoModelForVideoClassification, AutoFeatureExtractor
import subprocess
import time

def load_model(model_url: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"모델을 로드할 장치: {device}")
    model = AutoModelForVideoClassification.from_pretrained(model_url)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_url)
    model.to(device)
    model.eval()
    return model, feature_extractor

def get_top_prediction(model, predictions):
    top_class_index = predictions.argmax(-1).item()
    return model.config.id2label[top_class_index]

def process_video(
    video_path: Path, 
    model, 
    feature_extractor, 
    sampling_window_frames: int, 
    sliding_window_step_frames: int,
    num_frames_to_sample: int,
    progress_callback: Callable[[int, int], None],
    result_callback: Callable[[Dict], None],
    stop_checker: Callable[[], bool],
    frame_callback: Callable[[bytes], None]
) -> List[Dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise Exception("비디오 파일을 열 수 없습니다.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    device = model.device
    all_results = []
    
    current_pos_frame = 0
    while current_pos_frame + sampling_window_frames < total_frames:
        time.sleep(0.01)

        if stop_checker():
            print("추론 중지 신호 감지됨 (utils.py)")
            break

        sampling_start_frame = current_pos_frame
        overlay_start_frame = sampling_end_frame = current_pos_frame + sampling_window_frames
        overlay_end_frame = overlay_start_frame + sliding_window_step_frames
        
        frame_indices_to_sample = np.linspace(
            sampling_start_frame, sampling_end_frame - 1, num_frames_to_sample, dtype=int
        )
        
        batch_frames_rgb = []
        for idx in frame_indices_to_sample:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                batch_frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                break
        
        if len(batch_frames_rgb) < num_frames_to_sample:
            break

        try:
            inputs = feature_extractor(batch_frames_rgb, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = outputs.logits.softmax(dim=-1)
                prediction_label = get_top_prediction(model, predictions)

            result = {
                "video_name": video_path.name,
                "start_time": overlay_start_frame / fps,
                "end_time": overlay_end_frame / fps,
                "prediction_label": prediction_label,
                "start_frame": overlay_start_frame,
                "end_frame": overlay_end_frame
            }
            all_results.append(result)
            
            if result_callback:
                result_callback(result)

        except Exception as e:
            print(f"추론 중 에러 발생: {e}")

        # Now, get the frame to draw the overlay on
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos_frame) # Use current_pos_frame for overlay drawing
        ret, current_frame = cap.read()
        if ret:
            if prediction_label != "No Detection": # Only draw if there's an actual prediction
                label = prediction_label
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                text_x = (current_frame.shape[1] - text_size[0]) // 2
                text_y = 50
                cv2.putText(current_frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

            # Encode the frame to JPEG and send it via callback
            _, buffer = cv2.imencode('.jpg', current_frame)
            if frame_callback:
                frame_callback(buffer.tobytes())

        current_pos_frame += sliding_window_step_frames

        if progress_callback:
            progress_callback(min(current_pos_frame, total_frames), total_frames)

    cap.release()
    if progress_callback:
        progress_callback(total_frames, total_frames) # Ensure 100% progress is sent at the end of video processing
    return all_results

def create_overlay_video(video_path: Path, results: List[Dict], output_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
        '-c:v', 'libopenh264', '-pix_fmt', 'yuv420p', str(output_path)
    ]
    
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_idx = 0
    active_result = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        if active_result and frame_idx >= active_result['end_frame']: active_result = None
        if not active_result:
            for r in results:
                if r['start_frame'] <= frame_idx < r['end_frame']:
                    active_result = r
                    break
        if active_result:
            label = active_result['prediction_label']
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            text_x = (width - text_size[0]) // 2
            text_y = 50
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        try:
            proc.stdin.write(frame.tobytes())
        except (IOError, BrokenPipeError):
            print("ffmpeg 파이프가 닫혔습니다.")
            break
        frame_idx += 1
    cap.release()
    _, stderr_data = proc.communicate()
    if proc.returncode != 0:
        print(f"FFMPEG ERROR:\n{stderr_data.decode('utf-8', errors='ignore')}")

def create_results_csv(results: List[Dict], output_path: Path):
    if not results: return
    df = pd.DataFrame(results)
    df_to_save = df[['video_name', 'start_time', 'end_time', 'prediction_label']]
    df_to_save.to_csv(output_path, index=False, float_format='%.2f')

def get_video_info(video_path: Path) -> Dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return {}
    info = {
        "filename": video_path.name,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30))
    }
    cap.release()
    return info
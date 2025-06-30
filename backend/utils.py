# /backend/utils.py

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Callable
import pandas as pd
import torch
from transformers import AutoModelForVideoClassification, AutoFeatureExtractor
import time
import os
import traceback
from filelock import FileLock
import subprocess

def draw_overlay_text(frame, label_text: str, speed_text: str, width: int = None, height: int = None):
    """오버레이 텍스트를 프레임에 그리는 공통 함수"""
    if width is None:
        width = frame.shape[1]
    if height is None:
        height = frame.shape[0]
    
    # 폰트 및 스케일
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_font_scale = 0.7
    speed_font_scale = 0.5
    thickness = 2

    # 텍스트 크기 계산
    label_size, _ = cv2.getTextSize(label_text, font, label_font_scale, thickness)
    speed_size, _ = cv2.getTextSize(speed_text, font, speed_font_scale, thickness)

    # 여백 설정
    margin_x = 20
    margin_y = 20
    line_spacing = 10

    # 라벨 위치 계산 (우측 상단)
    label_x = width - label_size[0] - margin_x
    label_y = margin_y + label_size[1]

    # 속도 텍스트 위치 계산 (라벨 아래)
    speed_x = width - speed_size[0] - margin_x
    speed_y = label_y + line_spacing + speed_size[1]

    # 배경 박스 그리기
    overlay = frame.copy()
    alpha = 0.6  # 투명도

    # 라벨 배경
    cv2.rectangle(overlay, (label_x - 5, label_y - label_size[1] - 5), 
                 (label_x + label_size[0] + 5, label_y + 5), (0, 0, 0), -1)
    # 속도 텍스트 배경
    cv2.rectangle(overlay, (speed_x - 5, speed_y - speed_size[1] - 5), 
                 (speed_x + speed_size[0] + 5, speed_y + 5), (0, 0, 0), -1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 텍스트 그리기
    cv2.putText(frame, label_text, (label_x, label_y), font, label_font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.putText(frame, speed_text, (speed_x, speed_y), font, speed_font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

# 모델 캐시/다운로드만 담당 (실제 메모리 적재는 inference_manager에서)
def load_model(model_url: str, max_retries=3):
    """
    모델을 허깅페이스에서 다운로드(캐시)만 수행. 반드시 local_files_only=False로 동작.
    실제 추론에서는 사용하지 않음.
    """
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    lock_path = os.path.join(cache_dir, f"{model_url.replace('/', '_')}.lock")
    fail_flag_path = os.path.join(cache_dir, f"{model_url.replace('/', '_')}_fail.flag")
    with FileLock(lock_path):
        if os.path.exists(fail_flag_path):
            msg = f"모델 다운로드 실패 플래그 감지: {fail_flag_path}"
            print(msg)
            with open('/app/model_load_error.log', 'a') as f:
                f.write(msg + '\n')
            raise RuntimeError(msg)
        for attempt in range(1, max_retries + 1):
            try:
                # 반드시 local_files_only=False로 캐시 저장만 수행
                AutoModelForVideoClassification.from_pretrained(model_url, local_files_only=False)
                AutoFeatureExtractor.from_pretrained(model_url, local_files_only=False)
                return
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[모델 다운로드 실패 {attempt}/{max_retries}] {e}")
                with open('/app/model_load_error.log', 'a') as f:
                    f.write(f"[{attempt}] {tb}\n")
                if attempt == max_retries:
                    with open(fail_flag_path, 'w') as f:
                        f.write("fail")
                    raise
                else:
                    time.sleep(2)

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
    prediction_label = "No Detection"  # 기본값 설정
    inference_time_ms = 0
    inference_fps = 0
    
    while current_pos_frame < total_frames:
        # 진행률 업데이트를 더 자주 수행
        if progress_callback:
            progress_callback(min(current_pos_frame, total_frames), total_frames)

        if stop_checker():
            print("추론 중지 신호 감지됨 (utils.py)")
            break

        sampling_start_frame = current_pos_frame
        sampling_end_frame = min(current_pos_frame + sampling_window_frames, total_frames)
        overlay_start_frame = current_pos_frame
        overlay_end_frame = min(overlay_start_frame + sliding_window_step_frames, total_frames)
        
        actual_num_frames_to_sample = min(num_frames_to_sample, sampling_end_frame - sampling_start_frame)

        if actual_num_frames_to_sample <= 0:
            break

        frame_indices_to_sample = np.linspace(
            sampling_start_frame, sampling_end_frame - 1, actual_num_frames_to_sample, dtype=int
        )
        
        batch_frames_rgb = []
        for idx in frame_indices_to_sample:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                batch_frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                break
        
        if not batch_frames_rgb:
            current_pos_frame += sliding_window_step_frames
            continue

        try:
            # 추론 시작 시간 측정
            inference_start_time = time.time()
            
            # numpy 배열을 그대로 feature_extractor에 전달
            inputs = feature_extractor(batch_frames_rgb, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = outputs.logits.softmax(dim=-1)
                prediction_label = get_top_prediction(model, predictions)
            
            # 추론 종료 시간 측정 및 처리 속도 계산
            inference_end_time = time.time()
            inference_time_ms = (inference_end_time - inference_start_time) * 1000
            inference_fps = 1000 / inference_time_ms if inference_time_ms > 0 else 0

            result = {
                "video_name": video_path.name,
                "start_time": overlay_start_frame / fps,
                "end_time": overlay_end_frame / fps,
                "prediction_label": prediction_label,
                "start_frame": overlay_start_frame,
                "end_frame": overlay_end_frame,
                "inference_time_ms": round(inference_time_ms, 2),
                "inference_fps": round(inference_fps, 2)
            }
            all_results.append(result)
            
            if result_callback:
                result_callback(result)

        except Exception as e:
            print(f"추론 중 에러 발생: {e}")
            # 에러 발생 시 기본값 설정
            prediction_label = "No Detection"
            inference_time_ms = 0
            inference_fps = 0

        # Now, get the frame to draw the overlay on
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos_frame) # Use current_pos_frame for overlay drawing
        ret, current_frame = cap.read()
        if ret:
            if prediction_label != "No Detection": # Only draw if there's an actual prediction
                # 오버레이 텍스트 설정
                label_text = prediction_label
                speed_text = f"{inference_time_ms:.1f}ms ({inference_fps:.1f} FPS)"

                draw_overlay_text(current_frame, label_text, speed_text)

            # Encode the frame to JPEG and send it via callback (항상 전송)
            _, buffer = cv2.imencode('.jpg', current_frame)
            if frame_callback:
                frame_callback(buffer.tobytes())

        current_pos_frame += sliding_window_step_frames

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
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', 'faststart', str(output_path)
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
            # 오버레이 텍스트 설정
            label_text = active_result['prediction_label']
            speed_text = f"{active_result['inference_time_ms']:.1f}ms ({active_result['inference_fps']:.1f} FPS)"

            draw_overlay_text(frame, label_text, speed_text, width, height)

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
    df_to_save = df[['video_name', 'start_time', 'end_time', 'prediction_label', 'start_frame', 'end_frame', 'inference_time_ms', 'inference_fps']]
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
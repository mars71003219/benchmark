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
import asyncio

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
    model,
    feature_extractor,
    video_path: str,
    interval: int,
    infer_period: int,
    batch: int,
    save_dir: Path,
    inference_mode: str,
    annotation_data: Dict,
    progress_callback: Callable[[int, int], bool],
    realtime_frame_queue: asyncio.Queue,
    update_individual_event_callback: Callable[[Dict], None]
) -> List[Dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("비디오 파일을 열 수 없습니다.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    device = model.device
    all_results = []
    
    current_pos_frame = 0
    while current_pos_frame + interval < total_frames:
        if not progress_callback(current_pos_frame, total_frames):
            print("추론 중지 신호 감지됨 (utils.py)")
            break

        sampling_start_frame = current_pos_frame
        overlay_start_frame = sampling_end_frame = current_pos_frame + interval
        overlay_end_frame = overlay_start_frame + infer_period
        
        frame_indices_to_sample = np.linspace(
            sampling_start_frame, sampling_end_frame - 1, batch, dtype=int
        )
        
        batch_frames_rgb = []
        for idx in frame_indices_to_sample:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                batch_frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                break
        
        if len(batch_frames_rgb) < batch:
            break

        try:
            # 추론 시작 시간 측정
            inference_start_time = time.time()
            
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
                "video_name": Path(video_path).name,
                "start_time": overlay_start_frame / fps,
                "end_time": overlay_end_frame / fps,
                "prediction_label": prediction_label,
                "start_frame": overlay_start_frame,
                "end_frame": overlay_end_frame,
                "inference_time_ms": round(inference_time_ms, 2),
                "inference_fps": round(inference_fps, 2)
            }
            all_results.append(result)

            # 현재 프레임에 오버레이 추가 및 큐에 전송 (실시간 스트리밍용)
            cap.set(cv2.CAP_PROP_POS_FRAMES, overlay_start_frame)
            ret, current_frame = cap.read()
            if ret:
                # 오버레이 텍스트 설정
                label_text = prediction_label
                speed_text = f"{inference_time_ms:.1f}ms ({inference_fps:.1f} FPS)"

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
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                label_x = width - label_size[0] - margin_x
                label_y = margin_y + label_size[1]

                # 속도 텍스트 위치 계산 (라벨 아래)
                speed_x = width - speed_size[0] - margin_x
                speed_y = label_y + line_spacing + speed_size[1]

                # 배경 박스 그리기
                overlay_frame_copy = current_frame.copy()
                alpha = 0.6 # 투명도

                # 라벨 배경
                cv2.rectangle(overlay_frame_copy, (label_x - 5, label_y - label_size[1] - 5), (label_x + label_size[0] + 5, label_y + 5), (0, 0, 0), -1)
                # 속도 텍스트 배경
                cv2.rectangle(overlay_frame_copy, (speed_x - 5, speed_y - speed_size[1] - 5), (speed_x + speed_size[0] + 5, speed_y + 5), (0, 0, 0), -1)

                cv2.addWeighted(overlay_frame_copy, alpha, current_frame, 1 - alpha, 0, current_frame)

                # 텍스트 그리기
                cv2.putText(current_frame, label_text, (label_x, label_y), font, label_font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                cv2.putText(current_frame, speed_text, (speed_x, speed_y), font, speed_font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

                # 프레임을 JPEG로 인코딩하여 큐에 추가
                _, buffer = cv2.imencode('.jpg', current_frame)
                try:
                    realtime_frame_queue.put_nowait(buffer.tobytes())
                except asyncio.QueueFull:
                    pass # 큐가 가득 찼으면 최신 프레임을 위해 건너뜜

        except Exception as e:
            print(f"추론 중 에러 발생: {e}")

        current_pos_frame += infer_period

        # 각 추론 결과 발생 시 콜백 호출
        if result: # result가 존재할 경우에만 콜백 호출
            update_individual_event_callback(result)

    cap.release()
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
        '-c:v', 'libopenh264', '-pix_fmt', 'yuv420p', '-movflags', 'faststart', str(output_path)
    ]
    
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_idx = 0
    
    result_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        current_result = None
        if result_idx < len(results) and frame_idx >= results[result_idx]['start_frame']:
            current_result = results[result_idx]
            if frame_idx >= current_result['end_frame']: # Move to next result if current one ends
                result_idx += 1
                if result_idx < len(results) and frame_idx >= results[result_idx]['start_frame']:
                    current_result = results[result_idx]
                else:
                    current_result = None # No result for this frame yet
            
        if current_result:
            label_text = current_result['prediction_label']
            speed_text = f"{current_result['inference_time_ms']:.1f}ms ({current_result['inference_fps']:.1f} FPS)"

            font = cv2.FONT_HERSHEY_SIMPLEX
            label_font_scale = 0.7
            speed_font_scale = 0.5
            thickness = 2

            label_size, _ = cv2.getTextSize(label_text, font, label_font_scale, thickness)
            speed_size, _ = cv2.getTextSize(speed_text, font, speed_font_scale, thickness)

            margin_x = 20
            margin_y = 20
            line_spacing = 10

            label_x = width - label_size[0] - margin_x
            label_y = margin_y + label_size[1]
            speed_x = width - speed_size[0] - margin_x
            speed_y = label_y + line_spacing + speed_size[1]

            overlay_frame_copy = frame.copy()
            alpha = 0.6

            cv2.rectangle(overlay_frame_copy, (label_x - 5, label_y - label_size[1] - 5), (label_x + label_size[0] + 5, label_y + 5), (0, 0, 0), -1)
            cv2.rectangle(overlay_frame_copy, (speed_x - 5, speed_y - speed_size[1] - 5), (speed_x + speed_size[0] + 5, speed_y + 5), (0, 0, 0), -1)

            cv2.addWeighted(overlay_frame_copy, alpha, frame, 1 - alpha, 0, frame)

            cv2.putText(frame, label_text, (label_x, label_y), font, label_font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
            cv2.putText(frame, speed_text, (speed_x, speed_y), font, speed_font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

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
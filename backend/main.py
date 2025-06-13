# /backend/main.py

from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException, BackgroundTasks, Request, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from datetime import datetime
import asyncio
import aiofiles
from pathlib import Path
from typing import List, Dict, Callable, Optional
import functools
import cv2
import os
import subprocess
import sys
import base64
from collections import Counter
from pydantic import BaseModel

import psutil
from pynvml import *

from utils import (
    load_model, process_video, create_overlay_video,
    create_results_csv, get_video_info
)

import pandas as pd
import csv

# Request body for inference endpoint
class InferenceRequest(BaseModel):
    interval: int
    infer_period: int
    batch: int
    inference_mode: str
    annotation_data: Dict = {}

# Global queues for thread-safe communication between executor thread and main event loop
# Max size 1 to only keep the latest update/frame
realtime_frame_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
inference_state_queue: asyncio.Queue[Dict] = asyncio.Queue(maxsize=1)

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

stop_infer_flag = False
current_video_info = {}
model, feature_extractor = None, None

# Global WebSocket connection for real-time overlay frames
# This is a simplification. For a production app, you'd manage multiple client connections.
realtime_overlay_websocket: WebSocket | None = None

# Global counters for metrics (they are reset in reset_inference_state)
global_tp = 0
global_tn = 0
global_fp = 0
global_fn = 0
global_total_processed_clips = 0
global_correct_predictions = 0

# 전역 변수 추가
FINAL_RESULTS_FILE = RESULTS_DIR / "final_results.csv"

try:
    nvmlInit()
except NVMLError: pass

@app.on_event("shutdown")
def shutdown_event():
    try: nvmlShutdown()
    except NVMLError: pass

def initialize_final_results_file():
    """final_results.csv 파일을 초기화합니다."""
    if not FINAL_RESULTS_FILE.exists():
        with open(FINAL_RESULTS_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video_name', 'final_label', 'true_label', 'is_correct'])

def determine_final_label(video_name: str, prediction_labels: list) -> str:
    """
    비디오의 최종 라벨을 결정합니다.
    
    Args:
        video_name: 비디오 파일명
        prediction_labels: 해당 비디오의 모든 배치 샘플에 대한 예측 라벨 리스트
    
    Returns:
        str: 최종 결정된 라벨
    """
    if not prediction_labels:
        return None
        
    # 모든 라벨이 Non을 포함하는지 확인
    non_labels = [label for label in prediction_labels if label.startswith("Non")]
    
    if len(non_labels) == len(prediction_labels) and len(non_labels) > 0:
        # 모든 라벨이 Non을 포함하면, 가장 빈도가 높은 Non 라벨을 선택
        return Counter(non_labels).most_common(1)[0][0]
    else:
        # Non 라벨 외의 라벨이 있으면, 전체 라벨 중 가장 빈도가 높은 라벨을 선택
        return Counter(prediction_labels).most_common(1)[0][0]

def update_final_results(video_name: str, final_label: str, true_label: str = None):
    """
    final_results.csv 파일을 업데이트합니다.
    
    Args:
        video_name: 비디오 파일명
        final_label: 최종 결정된 라벨
        true_label: 실제 라벨 (어노테이션에서 가져온)
    """
    is_correct = final_label == true_label if true_label is not None else None
    
    with open(FINAL_RESULTS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([video_name, final_label, true_label, is_correct])

def calculate_metrics_from_final_results() -> dict:
    """
    final_results.csv 파일을 기반으로 메트릭을 계산합니다.
    
    Returns:
        dict: 계산된 메트릭 (precision, recall, f1_score)
    """
    if not FINAL_RESULTS_FILE.exists():
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
    df = pd.read_csv(FINAL_RESULTS_FILE)
    if df.empty:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
    tp = len(df[(df['final_label'].str.startswith('fight')) & (df['is_correct'] == True)])
    tn = len(df[(~df['final_label'].str.startswith('fight')) & (df['is_correct'] == True)])
    fp = len(df[(df['final_label'].str.startswith('fight')) & (df['is_correct'] == False)])
    fn = len(df[(~df['final_label'].str.startswith('fight')) & (df['is_correct'] == False)])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1_score, 2),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

def reset_inference_state():
    """추론 상태를 초기화합니다."""
    global current_video_info, stop_infer_flag, global_tp, global_tn, global_fp, global_fn, global_total_processed_clips, global_correct_predictions
    current_video_info = {
        "status": "idle",
        "current_video": None,
        "progress": 0,
        "processed_videos": 0,
        "total_videos": 0,
        "events": [],
        "metrics": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
        "cumulative_accuracy": 0.0,
        "is_inferencing": False,
        "error": None
    }
    stop_infer_flag = False
    global_tp = 0
    global_tn = 0
    global_fp = 0
    global_fn = 0
    global_total_processed_clips = 0
    global_correct_predictions = 0
    
    # final_results.csv 파일 초기화
    initialize_final_results_file()

def stop_checker():
    return stop_infer_flag

# [수정] 이 함수는 이제 동기적으로 실행됨 (run_in_executor에서 호출되므로)
def process_all_videos_sync(interval, infer_period, batch, save_dir, inference_mode: str, annotation_data: Dict):
    global current_video_info, stop_infer_flag, global_tp, global_tn, global_fp, global_fn, global_total_processed_clips, global_correct_predictions
    try:
        # 추론이 시작되었음을 알리는 초기 상태 전송
        current_video_info["is_inferencing"] = True
        try:
            inference_state_queue.put_nowait(current_video_info.copy())
        except asyncio.QueueFull:
            pass

        # Get all video files
        video_files = sorted([f for f in UPLOAD_DIR.glob("*") if f.is_file()])
        if not video_files:
            print("처리할 비디오 파일이 없습니다.")
            current_video_info["status"] = "completed"
            current_video_info["is_inferencing"] = False
            try:
                inference_state_queue.put_nowait(current_video_info.copy())
            except asyncio.QueueFull:
                pass
            return

        # Process each video
        for video_path in video_files:
            if stop_infer_flag:
                print("추론 중지 요청 감지됨")
                break

            current_video_info["current_video"] = str(video_path.name)
            current_video_info["progress"] = 0
            current_video_info["status"] = "processing"
            try:
                inference_state_queue.put_nowait(current_video_info.copy())
            except asyncio.QueueFull:
                pass

            def progress_callback(done, total):
                if stop_infer_flag:
                    return False
                current_video_info["progress"] = int((done / total) * 100)
                try:
                    inference_state_queue.put_nowait(current_video_info.copy())
                except asyncio.QueueFull:
                    pass
                return True

            def update_individual_event_callback(result: Dict):
                current_video_info['events'].append(result)
                try:
                    inference_state_queue.put_nowait(current_video_info.copy())
                except asyncio.QueueFull:
                    pass

            try:
                video_results = process_video(
                    model=model,
                    feature_extractor=feature_extractor,
                    video_path=str(video_path),
                    interval=interval,
                    infer_period=infer_period,
                    batch=batch,
                    save_dir=save_dir,
                    inference_mode=inference_mode,
                    annotation_data=annotation_data,
                    progress_callback=progress_callback,
                    realtime_frame_queue=realtime_frame_queue,
                    update_individual_event_callback=update_individual_event_callback
                )
                
                if video_results:
                    # 비디오의 모든 배치 샘플에 대한 예측 라벨 수집
                    prediction_labels = [r['prediction_label'] for r in video_results]
                    
                    # 최종 라벨 결정
                    final_label = determine_final_label(video_path.name, prediction_labels)
                    
                    if final_label:
                        # 어노테이션에서 실제 라벨 가져오기
                        true_label = None
                        video_name_stem = Path(video_path).stem
                        if video_name_stem in annotation_data:
                            for ann in annotation_data[video_name_stem]:
                                if 'label' in ann:
                                    true_label = ann['label']
                                    break
                        
                        # final_results.csv 업데이트
                        update_final_results(video_path.name, final_label, true_label)
                        
                        # 메트릭 계산 및 업데이트
                        metrics = calculate_metrics_from_final_results()
                        current_video_info["metrics"] = metrics
                        
                        # 상태 업데이트 및 전송
                        try:
                            inference_state_queue.put_nowait(current_video_info.copy())
                        except asyncio.QueueFull:
                            pass
                
            except Exception as e:
                print(f"비디오 처리 중 오류 발생: {str(e)}")
                current_video_info["status"] = "error"
                current_video_info["error"] = str(e)
                try:
                    inference_state_queue.put_nowait(current_video_info.copy())
                except asyncio.QueueFull:
                    pass
                continue

            current_video_info["progress"] = 100
            current_video_info["status"] = "completed"
            current_video_info["processed_videos"] += 1
            current_video_info["total_videos"] = len(video_files)
            try:
                inference_state_queue.put_nowait(current_video_info.copy())
            except asyncio.QueueFull:
                pass

    except Exception as e:
        print(f"비디오 처리 중 오류 발생: {str(e)}")
        current_video_info["status"] = "error"
        current_video_info["error"] = str(e)
        try:
            inference_state_queue.put_nowait(current_video_info.copy())
        except asyncio.QueueFull:
            pass
    finally:
        current_video_info["is_inferencing"] = False
        try:
            inference_state_queue.put_nowait(current_video_info.copy())
        except asyncio.QueueFull:
            pass

@app.post("/infer")
async def start_inference_endpoint(request: InferenceRequest):
    global stop_infer_flag, model, feature_extractor
    if model is None or feature_extractor is None:
        raise HTTPException(status_code=400, detail="모델이 로드되지 않았습니다.")
    
    stop_infer_flag = True # Signal to stop current inference
    # Give a moment for the previous inference to stop. A more robust solution might wait for a stop confirmation.
    await asyncio.sleep(0.1)
    stop_infer_flag = False # Reset flag for new inference

    # Reset global counters and inference state for new inference
    reset_inference_state()
    print("[start_inference_endpoint] 추론 시작 요청 받음. 초기 상태 초기화 완료.")

    # Run inference in a background thread to not block the main event loop
    # Pass the annotation_data to the processing function
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None, functools.partial(
            process_all_videos_sync,
            interval=request.interval, infer_period=request.infer_period, batch=request.batch,
            save_dir=RESULTS_DIR, inference_mode=request.inference_mode, annotation_data=request.annotation_data
        )
    )
    print("[start_inference_endpoint] process_all_videos_sync가 백그라운드에서 실행되도록 예약됨.")

    return JSONResponse(content={
        "message": "추론 시작됨", "status": "inferring",
        "inference_mode": request.inference_mode
    })

@app.post("/model")
async def set_model_endpoint(request: Request):
    global model, feature_extractor
    data = await request.json()
    model_id = data.get("model_id")
    if not model_id: raise HTTPException(400, "model_id가 필요합니다.")
    model, feature_extractor = load_model(model_id)
    print(f"[set_model_endpoint] 모델 {model_id} 로드 완료.")
    return {"message": "모델 로드 완료"}

@app.post("/upload")
async def upload_videos_endpoint(request: Request):
    uploaded = []
    form = await request.form()
    files = form.getlist("files")
    paths = form.getlist("paths")

    for i, file in enumerate(files):
        relative_path = Path(paths[i])
        full_path = UPLOAD_DIR / Path(relative_path).name
        # full_path.parent.mkdir(parents=True, exist_ok=True) # No need to create subdirectories
        
        async with aiofiles.open(full_path, 'wb') as f:
            await f.write(await file.read())
        
        metadata = get_video_metadata(str(full_path))
        uploaded.append({"name": str(relative_path), "size": full_path.stat().st_size, "duration": metadata['duration'] if metadata else None})
    print(f"[upload_videos_endpoint] {len(uploaded)}개 파일 업로드 완료.")
    return {"files": uploaded}

@app.get("/uploads")
async def get_uploads_endpoint():
    files = []
    for root, _, filenames in os.walk(UPLOAD_DIR):
        for filename in filenames:
            full_path = Path(root) / filename
            if full_path.is_file():
                relative_path = full_path.relative_to(UPLOAD_DIR)
                metadata = get_video_metadata(str(full_path))
                files.append({"name": str(relative_path), "size": full_path.stat().st_size, "duration": metadata['duration'] if metadata else None})
    print(f"[get_uploads_endpoint] {len(files)}개 업로드 파일 조회 완료.")
    return {"files": sorted(files, key=lambda x: x['name'])}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[websocket_endpoint] WebSocket 연결 수락됨.")
    try:
        # 연결이 설정될 때 현재 전역 상태의 초기 스냅샷을 한 번 보냅니다.
        await websocket.send_json(current_video_info.copy())
        print("[websocket_endpoint] 초기 current_video_info 전송됨.")

        while True:
            try:
                # 큐에서 다음 상태 업데이트를 기다립니다. 이는 항목이 사용 가능해질 때까지 차단됩니다.
                latest_state = await inference_state_queue.get()
                await websocket.send_json(latest_state)
                print(f"[websocket_endpoint] 큐에서 상태 업데이트 가져와 클라이언트에 전송됨: {latest_state.get('current_progress', '?')}%")
            except WebSocketDisconnect:
                print("[websocket_endpoint] WebSocket 연결이 끊어졌습니다.")
                break
            except Exception as e:
                print(f"[websocket_endpoint] WebSocket 처리 중 오류 발생: {e}")
                try:
                    await websocket.send_json({"error": str(e)})
                except:
                    print("[websocket_endpoint] 오류 메시지 전송 실패.")
                    break
    except Exception as e:
        print(f"[websocket_endpoint] WebSocket 메인 루프 오류: {e}")
    finally:
        print("[websocket_endpoint] WebSocket 연결 종료.")
        try:
            await websocket.close()
        except:
            print("[websocket_endpoint] WebSocket 닫기 실패.")
            pass

@app.post("/stop_infer")
async def stop_infer_endpoint():
    global stop_infer_flag
    stop_infer_flag = True
    print("[stop_infer_endpoint] 추론 중지 요청 받음.")
    return {"message": "추론 중지 요청"}

@app.delete("/upload/{filename}")
async def delete_file_endpoint(filename: str):
    path = UPLOAD_DIR / filename
    if path.exists(): path.unlink()
    print(f"[delete_file_endpoint] 파일 {filename} 삭제 완료.")
    return {"message": f"{filename} 삭제 완료"}

@app.delete("/uploads")
async def delete_all_files_endpoint():
    for f in UPLOAD_DIR.glob("*"):
        if f.is_file(): f.unlink()
    reset_inference_state()
    print("[delete_all_files_endpoint] 모든 업로드 파일 삭제 및 추론 상태 초기화 완료.")
    return {"message": "모든 업로드 파일 삭제 완료"}

@app.post("/delete_specific_uploads")
async def delete_specific_uploads_endpoint(request: Request):
    data = await request.json()
    filenames = data.get("filenames", [])
    
    deleted_count = 0
    for filename in filenames:
        # Ensure we only delete files directly in UPLOAD_DIR or its subdirectories
        # using os.path.join for safety against directory traversal attacks
        file_path = UPLOAD_DIR / filename
        if file_path.exists() and file_path.is_file():
            try:
                file_path.unlink()
                deleted_count += 1
                print(f"[delete_specific_uploads_endpoint] 파일 {filename} 삭제 성공.")
            except Exception as e:
                print(f"[delete_specific_uploads_endpoint] 파일 삭제 실패 ({filename}): {e}")
    
    print(f"[delete_specific_uploads_endpoint] {deleted_count}개 파일 삭제 완료.")
    return {"message": f"{deleted_count}개 파일 삭제 완료"}

@app.get("/system_info")
async def get_system_info_endpoint():
    info = {"cpu": None, "ram": None, "ram_used_mb": None, "ram_total_mb": None, "gpu": None, "gpuMem": None, "gpuUtil": None}
    try:
        info["cpu"] = psutil.cpu_percent(0.1)
        vmem = psutil.virtual_memory()
        info["ram"], info["ram_used_mb"], info["ram_total_mb"] = vmem.percent, vmem.used / 1e6, vmem.total / 1e6
    except Exception: pass
    try:
        if nvmlDeviceGetCount() > 0:
            handle = nvmlDeviceGetHandleByIndex(0)
            info["gpu"] = nvmlDeviceGetName(handle)
            mem = nvmlDeviceGetMemoryInfo(handle)
            info["gpuMem"] = f"{mem.used/1e6:.0f}MB / {mem.total/1e6:.0f}MB"
            info["gpuUtil"] = nvmlDeviceGetUtilizationRates(handle).gpu
    except NVMLError: pass
    print("[get_system_info_endpoint] 시스템 정보 조회 완료.")
    return info

@app.get("/results/videos")
async def get_result_videos_endpoint():
    videos = sorted([f.name for f in RESULTS_DIR.glob("*_overlay.mp4")])
    print(f"[get_result_videos_endpoint] {len(videos)}개 결과 비디오 조회 완료.")
    return {"videos": videos}

@app.get("/video/{video_id}/overlay")
async def get_overlay_video_endpoint(video_id: str):
    try:
        path = RESULTS_DIR / f"{Path(video_id).stem}_overlay.mp4"
        if not path.exists():
            print(f"[get_overlay_video_endpoint] 비디오 파일을 찾을 수 없음: {path}")
            raise HTTPException(404, "파일 없음")
        
        # Get file size to set Content-Length header
        file_size = os.path.getsize(path)

        print(f"[get_overlay_video_endpoint] 비디오 스트리밍 시작: {path}")
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=path.name,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size)
            }
        )
    except Exception as e:
        print(f"[get_overlay_video_endpoint] 비디오 스트리밍 중 에러 발생: {str(e)}")
        raise HTTPException(500, f"비디오 스트리밍 실패: {str(e)}")

@app.get("/results.csv")
async def get_results_csv_endpoint():
    path = RESULTS_DIR / "results.csv"
    if not path.exists():
        print("[get_results_csv_endpoint] results.csv 파일을 찾을 수 없음.")
        raise HTTPException(404, "파일 없음")
    print("[get_results_csv_endpoint] results.csv 조회 완료.")
    return FileResponse(path)

@app.get("/video_metadata/{video_path}")
async def get_video_metadata_endpoint(video_path: str):
    try:
        metadata = get_video_metadata(video_path)
        if metadata:
            print(f"[get_video_metadata_endpoint] 비디오 메타데이터 조회 완료: {video_path}")
            return metadata
        else:
            print(f"[get_video_metadata_endpoint] 메타데이터 가져오기 실패: {video_path}")
            raise HTTPException(500, "메타데이터 가져오기 실패")
    except Exception as e:
        print(f"[get_video_metadata_endpoint] 메타데이터 가져오기 실패: {str(e)}")
        raise HTTPException(500, "메타데이터 가져오기 실패")

def get_video_metadata(video_path):
    """비디오 파일의 메타데이터를 가져옵니다."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[get_video_metadata] 비디오 파일 열기 실패: {video_path}")
            return None
        
        # 비디오 정보 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        # 파일 크기 가져오기
        file_size = os.path.getsize(video_path)
        file_size_mb = file_size / (1024 * 1024)  # MB로 변환
        
        # 파일 생성 시간 가져오기
        creation_time = os.path.getctime(video_path)
        creation_date = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
        
        cap.release()
        print(f"[get_video_metadata] 비디오 메타데이터 성공적으로 가져옴: {video_path}")
        
        return {
            'fps': round(fps, 2),
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': round(duration, 2),
            'file_size_mb': round(file_size_mb, 2),
            'creation_date': creation_date
        }
    except Exception as e:
        print(f"[get_video_metadata] 메타데이터 가져오기 실패: {str(e)}")
        return None

@app.websocket("/ws/realtime_overlay")
async def websocket_realtime_overlay_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[websocket_realtime_overlay_endpoint] 실시간 오버레이 WebSocket 연결 수락됨.")
    try:
        while True:
            frame_data = await realtime_frame_queue.get() # Blocking get until a frame is available
            encoded_frame = base64.b64encode(frame_data).decode('utf-8')
            await websocket.send_text(encoded_frame)
            print("[websocket_realtime_overlay_endpoint] 실시간 프레임 전송됨.")
    except Exception as e:
        print(f"[websocket_realtime_overlay_endpoint] 실시간 오버레이 WebSocket 오류: {e}")
    finally:
        print("[websocket_realtime_overlay_endpoint] 실시간 오버레이 WebSocket 연결 종료.")

if __name__ == "__main__":
    import uvicorn
    reset_inference_state()
    print("[main] 초기 추론 상태 초기화 완료.")
    uvicorn.run(app, host="0.0.0.0", port=10000)



# /backend/main.py

import logging
logging.getLogger().setLevel(logging.CRITICAL)
for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "starlette"):
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

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
import csv
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import torch
import time
import multiprocessing as mp

# pynvml 관련 import를 try-except로 감싸서 선택적으로 import
try:
    from pynvml import *
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    # 더미 함수들 정의
    def nvmlInit(): pass
    def nvmlShutdown(): pass
    def nvmlDeviceGetCount(): return 0
    def nvmlDeviceGetHandleByIndex(index): return None
    def nvmlDeviceGetName(handle): return "N/A"
    def nvmlDeviceGetMemoryInfo(handle): return type('obj', (object,), {'used': 0, 'total': 0})()
    def nvmlDeviceGetUtilizationRates(handle): return type('obj', (object,), {'gpu': 0})()
    class NVMLError(Exception): pass

from utils import (
    load_model, process_video, create_overlay_video,
    create_results_csv, get_video_info
)

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
RESULTS_DIR = Path("/aivanas/raw/surveillance/action/eval_results/temp_results")
SAVE_RESULTS_DIR = Path("/aivanas/raw/surveillance/action/eval_results/save_results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SAVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# === 멀티프로세스 기반으로 변경됨 - 기존 글로벌 변수들은 shared_state로 대체 ===
stop_infer_flag = False  # 추가: stop_infer_flag 정의
model, feature_extractor = None, None
current_model_id = None  # 현재 로드된 모델 id를 저장

# Global WebSocket connection for real-time overlay frames
# This is a simplification. For a production app, you'd manage multiple client connections.
realtime_overlay_websocket: WebSocket | None = None

# === 기존 글로벌 변수들은 shared_state로 대체됨 ===
# global_tp = 0
# global_tn = 0
# global_fp = 0
# global_fn = 0
# global_total_processed_clips = 0
# global_correct_predictions = 0

# pause_infer_flag = False
# resume_infer_flag = False
# paused_video_name = None

# NAS 폴더 복사 진행 상태 추적
nas_copy_progress = {
    "is_copying": False,
    "total_files": 0,
    "copied_files": 0,
    "current_file": None,
    "errors": []
}

# NAS 경로 설정
# 환경변수에서 읽어오거나 기본값 사용
NAS_BASE_PATH = os.getenv('NAS_BASE_PATH', '/home/hsnam')
NAS_TARGET_PATH = os.getenv('NAS_TARGET_PATH', '/aivanas')

print(f"NAS 경로 설정: {NAS_BASE_PATH} -> {NAS_TARGET_PATH}")

if NVML_AVAILABLE:
    try:
        nvmlInit()
    except NVMLError: 
        pass

logging.basicConfig(
    level=logging.CRITICAL,  # CRITICAL만 출력
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# throttled_push_state 함수 추가
last_push_time = 0

def throttled_push_state(state, min_interval=1.0):
    global last_push_time
    now = time.time()
    if now - last_push_time > min_interval:
        push_state(state)
        last_push_time = now

# reset_inference_state 함수 추가
def reset_inference_state():
    global stop_infer_flag
    stop_infer_flag = False
    # shared_state 초기화
    if 'shared_state' in globals():
        shared_state.clear()
        shared_state.update({
            "total_videos": 0,
            "processed_videos": 0,
            "current_video": None,
            "current_progress": 0,
            "events": [],
            "per_video_progress": {},
            "is_inferencing": False,
            "cumulative_accuracy": 0.0,
            "metrics": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0},
        })
    # Clear queues when state is reset for new inference
    while not realtime_frame_queue.empty():
        try: 
            realtime_frame_queue.get_nowait()
        except asyncio.QueueEmpty: 
            pass
    while not inference_state_queue.empty():
        try: 
            inference_state_queue.get_nowait()
        except asyncio.QueueEmpty: 
            pass

@app.on_event("shutdown")
def shutdown_event():
    global model, feature_extractor, current_model_id
    try:
        try:
            del model
        except Exception:
            pass
        try:
            del feature_extractor
        except Exception:
            pass
        model = None
        feature_extractor = None
        current_model_id = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    try: 
        if NVML_AVAILABLE:
            nvmlShutdown()
    except NVMLError: 
        pass

# === 기존 함수들은 멀티프로세스 기반으로 대체됨 ===
# def process_all_videos_sync(interval, infer_period, batch, save_dir, inference_mode: str, annotation_data: Dict, min_consecutive: int = 3):
#     # === 이 함수는 멀티프로세스 워커로 대체됨 ===
#     pass

def reset_nas_copy_progress():
    global nas_copy_progress
    nas_copy_progress = {
        "is_copying": False,
        "total_files": 0,
        "copied_files": 0,
        "current_file": None,
        "errors": []
    }

def copy_nas_folder_sync(nas_folder_path: str):
    """NAS 폴더의 모든 비디오 파일을 uploads로 복사하는 동기 함수 (이미 있는 파일은 건너뜀)"""
    global nas_copy_progress
    reset_nas_copy_progress()
    nas_copy_progress["is_copying"] = True
    skipped_files = []
    try:
        # 경로 정규화
        nas_path = Path(nas_folder_path).resolve()
        # print(f"정규화된 경로: {nas_path}")
        # 경로가 존재하는지 확인
        if not nas_path.exists():
            nas_copy_progress["errors"].append(f"경로가 존재하지 않습니다: {nas_folder_path}")
            return
        # 경로가 디렉토리인지 확인
        if not nas_path.is_dir():
            nas_copy_progress["errors"].append(f"지정된 경로가 디렉토리가 아닙니다: {nas_folder_path}")
            return
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        video_files = []
        for ext in video_extensions:
            video_files.extend(nas_path.rglob(f"*{ext}"))
            video_files.extend(nas_path.rglob(f"*{ext.upper()}"))
        video_files = list(set(video_files))
        video_files.sort()
        nas_copy_progress["total_files"] = len(video_files)
        if nas_copy_progress["total_files"] == 0:
            nas_copy_progress["errors"].append("선택된 폴더에서 비디오 파일을 찾을 수 없습니다.")
            return
        # print(f"NAS 폴더에서 {nas_copy_progress['total_files']}개의 비디오 파일을 찾았습니다.")
        for i, video_file in enumerate(video_files):
            if not nas_copy_progress["is_copying"]:
                # print("NAS 폴더 복사가 취소되었습니다.")
                break
            try:
                nas_copy_progress["current_file"] = str(video_file)
                target_filename = video_file.name
                target_path = UPLOAD_DIR / target_filename
                counter = 1
                original_target_path = target_path
                while target_path.exists():
                    name_without_ext = original_target_path.stem
                    ext = original_target_path.suffix
                    target_filename = f"{name_without_ext}_{counter}{ext}"
                    target_path = UPLOAD_DIR / target_filename
                    counter += 1
                if target_path.exists():
                    skipped_files.append(str(target_path.name))
                    # print(f"이미 존재하여 건너뜀: {target_path.name}")
                    nas_copy_progress["copied_files"] += 1
                    continue
                shutil.copy2(video_file, target_path)
                nas_copy_progress["copied_files"] += 1
                # print(f"복사 완료: {video_file.name} -> {target_filename} ({i+1}/{nas_copy_progress['total_files']})")
            except Exception as e:
                error_msg = f"파일 복사 실패 ({video_file.name}): {str(e)}"
                nas_copy_progress["errors"].append(error_msg)
                # print(error_msg)
        if skipped_files:
            nas_copy_progress["errors"].append(f"이미 존재하여 건너뛴 파일: {', '.join(skipped_files)}")
        # if nas_copy_progress["is_copying"]:
        #     print(f"NAS 폴더 복사 완료: {nas_copy_progress['copied_files']}/{nas_copy_progress['total_files']} 파일")
        # else:
        #     print(f"NAS 폴더 복사 취소됨: {nas_copy_progress['copied_files']}/{nas_copy_progress['total_files']} 파일 복사됨")
    except Exception as e:
        error_msg = f"NAS 폴더 처리 중 오류 발생: {str(e)}"
        nas_copy_progress["errors"].append(error_msg)
        # print(error_msg)
    finally:
        nas_copy_progress["is_copying"] = False

def stop_checker():
    return stop_infer_flag

def remove_video_results_from_csv(csv_path, video_name):
    if not csv_path.exists():
        return
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df['video_name'] != video_name]
    df.to_csv(csv_path, index=False)

# === 기존 함수들은 멀티프로세스 기반으로 대체됨 ===
# def process_all_videos_sync(interval, infer_period, batch, save_dir, inference_mode: str, annotation_data: Dict, min_consecutive: int = 3):
#     # === 이 함수는 멀티프로세스 워커로 대체됨 ===
#     pass

@app.post("/infer")
async def start_inference_endpoint(request: Request):
    if not model: 
        raise HTTPException(400, "모델이 로드되지 않았습니다.")
    if not any(UPLOAD_DIR.iterdir()): 
        raise HTTPException(400, "업로드된 비디오가 없습니다.")
    
    # 상태 초기화
    reset_inference_state()
    
    data = await request.json()
    interval = data.get("interval", 90)
    infer_period = data.get("infer_period", 30)
    batch = data.get("batch", 16)
    inference_mode = data.get("inference_mode", "default")
    annotation_data = data.get("annotation_data", {})
    min_consecutive = data.get("min_consecutive", 3)
    model_id = current_model_id

    video_files = sorted([p for p in UPLOAD_DIR.glob('*') if p.suffix in ['.mp4', '.avi', '.mov']])
    
    # shared_state 초기화
    shared_state.clear()
    shared_state.update({
        'is_inferencing': True,
        'total_videos': len(video_files),
        'processed_videos': 0,
        'events': [],
        'current_video': None,
        'current_progress': 0,
        'per_video_progress': {},
        'cumulative_accuracy': 0.0,
        'metrics': {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0},
    })
    
    for video_path in video_files:
        task = {
            'video_path': video_path,
            'interval': interval,
            'infer_period': infer_period,
            'batch': batch,
            'model_id': model_id,
            'inference_mode': inference_mode,
            'annotation_data': annotation_data,
            'min_consecutive': min_consecutive
        }
        inference_task_queue.put(task)
        shared_state['events'].append({"type": "start", "video": video_path.name, "timestamp": datetime.now().isoformat()})
    
    return {"message": "추론이 백그라운드에서 시작되었습니다."}

@app.post("/model")
async def set_model_endpoint(request: Request):
    global model, feature_extractor, current_model_id
    data = await request.json()
    model_id = data.get("model_id")
    if not model_id: 
        raise HTTPException(400, "model_id가 필요합니다.")
    model, feature_extractor = load_model(model_id)
    current_model_id = model_id  # 모델 id 저장
    return {"message": "모델 로드 완료"}

@app.get("/current_model")
async def get_current_model_endpoint():
    global current_model_id
    return {"model_id": current_model_id}

@app.post("/upload")
async def upload_videos_endpoint(request: Request):
    uploaded = []
    skipped = []
    form = await request.form()
    files = form.getlist("files")
    paths = form.getlist("paths")

    for i, file in enumerate(files):
        if isinstance(file, str):
            # paths[i]가 파일 경로인 경우
            relative_path = Path(paths[i])
            full_path = UPLOAD_DIR / Path(relative_path).name
            if full_path.exists():
                skipped.append(str(relative_path))
                continue
            # 파일이 이미 존재하는 경우 스킵
            if full_path.exists():
                skipped.append(str(relative_path))
                continue
        else:
            # UploadFile 객체인 경우
            relative_path = Path(paths[i])
            full_path = UPLOAD_DIR / Path(relative_path).name
            if full_path.exists():
                skipped.append(str(relative_path))
                continue
            async with aiofiles.open(full_path, 'wb') as f:
                await f.write(await file.read())
        
        metadata = get_video_metadata(str(full_path))
        uploaded.append({"name": str(relative_path), "size": full_path.stat().st_size, "duration": metadata['duration'] if metadata else None})
    return {"files": uploaded, "skipped": skipped}

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
    return {"files": sorted(files, key=lambda x: x['name'])}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        if shared_state is None:
            await websocket.send_json({"error": "서버 초기화 중입니다. 잠시 후 다시 시도해 주세요."})
            return
        await websocket.send_json(dict(shared_state))
        while True:
            await websocket.send_json(dict(shared_state))
            # ping/pong keepalive
            try:
                await websocket.send_text("ping")
                pong = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                if pong != "pong":
                    break
            except asyncio.TimeoutError:
                break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.critical("WebSocket disconnected")
    except Exception as e:
        logger.critical(e)
    finally:
        logger.critical("WebSocket connection closed.")

@app.post("/stop_infer")
async def stop_infer_endpoint():
    global stop_infer_flag
    stop_infer_flag = True
    # 결과 파일 이동
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = SAVE_RESULTS_DIR / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    for f in RESULTS_DIR.glob("*"):
        shutil.move(str(f), str(save_dir / f.name))
    return {"message": "추론 중지 및 결과 이동 완료"}

@app.delete("/upload/{filename}")
async def delete_file_endpoint(filename: str):
    path = UPLOAD_DIR / filename
    if path.exists(): 
        path.unlink()
    return {"message": f"{filename} 삭제 완료"}

@app.delete("/uploads")
async def delete_all_files_endpoint():
    for f in UPLOAD_DIR.glob("*"):
        if f.is_file(): 
            f.unlink()
    reset_inference_state()
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
            except Exception as e:
                print(f"파일 삭제 실패 ({filename}): {e}")
    
    return {"message": f"{deleted_count}개 파일 삭제 완료"}

@app.get("/system_info")
async def get_system_info_endpoint():
    info = {"cpu": None, "ram": None, "ram_used_mb": None, "ram_total_mb": None, "gpu": None, "gpuMem": None, "gpuUtil": None}
    try:
        info["cpu"] = psutil.cpu_percent(0.1)
        vmem = psutil.virtual_memory()
        info["ram"], info["ram_used_mb"], info["ram_total_mb"] = vmem.percent, vmem.used / 1e6, vmem.total / 1e6
    except Exception: 
        pass
    try:
        if NVML_AVAILABLE and nvmlDeviceGetCount() > 0:
            handle = nvmlDeviceGetHandleByIndex(0)
            info["gpu"] = nvmlDeviceGetName(handle)
            mem = nvmlDeviceGetMemoryInfo(handle)
            info["gpuMem"] = f"{mem.used/1e6:.0f}MB / {mem.total/1e6:.0f}MB"
            info["gpuUtil"] = nvmlDeviceGetUtilizationRates(handle).gpu
    except NVMLError: 
        pass
    return info

@app.get("/results/videos")
async def get_result_videos_endpoint():
    return {"videos": sorted([f.name for f in RESULTS_DIR.glob("*_overlay.mp4")])}

@app.get("/video/{video_id}/overlay")
async def get_overlay_video_endpoint(video_id: str):
    try:
        path = RESULTS_DIR / f"{Path(video_id).stem}_overlay.mp4"
        if not path.exists():
            print(f"비디오 파일을 찾을 수 없음: {path}")
            raise HTTPException(404, "파일 없음")
        
        # Get file size to set Content-Length header
        file_size = os.path.getsize(path)

        print(f"비디오 스트리밍 시작: {path}")
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
        print(f"비디오 스트리밍 중 에러 발생: {str(e)}")
        raise HTTPException(500, f"비디오 스트리밍 실패: {str(e)}")

@app.get("/results.csv")
async def get_results_csv_endpoint():
    path = RESULTS_DIR / "results.csv"
    if not path.exists(): 
        raise HTTPException(404, "파일 없음")
    return FileResponse(path)

@app.get("/video_metadata/{video_path}")
async def get_video_metadata_endpoint(video_path: str):
    try:
        metadata = get_video_metadata(video_path)
        if metadata:
            return metadata
        else:
            raise HTTPException(500, "메타데이터 가져오기 실패")
    except Exception as e:
        print(f"메타데이터 가져오기 실패: {str(e)}")
        raise HTTPException(500, "메타데이터 가져오기 실패")

def get_video_metadata(video_path):
    """비디오 파일의 메타데이터를 가져옵니다."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
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
        print(f"메타데이터 가져오기 실패: {str(e)}")
        return None

@app.websocket("/ws/realtime_overlay")
async def websocket_realtime_overlay_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # 재접속 시 최신 프레임 즉시 전송
        if not realtime_frame_queue.empty():
            latest_frame = None
            while not realtime_frame_queue.empty():
                latest_frame = await realtime_frame_queue.get()
            if latest_frame is not None:
                encoded_frame = base64.b64encode(latest_frame).decode('utf-8')
                await websocket.send_text(encoded_frame)
        while True:
            frame_data = await realtime_frame_queue.get() # Blocking get until a frame is available
            encoded_frame = base64.b64encode(frame_data).decode('utf-8')
            await websocket.send_text(encoded_frame)
            # ping/pong keepalive
            try:
                await websocket.send_text("ping")
                pong = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                if pong != "pong":
                    break
            except asyncio.TimeoutError:
                break
    except WebSocketDisconnect:
        logger.critical("실시간 오버레이 WebSocket 연결이 클라이언트에 의해 종료되었습니다.")
    except Exception as e:
        logger.critical(e)
    finally:
        logger.critical("실시간 오버레이 WebSocket 연결 종료.")

@app.post("/pause_infer")
async def pause_infer_endpoint():
    global pause_infer_flag
    pause_infer_flag = True
    return {"message": "추론 일시정지"}

@app.post("/resume_infer")
async def resume_infer_endpoint():
    global pause_infer_flag, resume_infer_flag
    pause_infer_flag = False
    resume_infer_flag = True
    return {"message": "추론 재개"}

@app.get("/current_inference_state")
async def get_current_inference_state_endpoint():
    return dict(shared_state)

@app.post("/unload_model")
async def unload_model_endpoint():
    global model, feature_extractor, current_model_id
    try:
        try:
            del model
        except Exception:
            pass
        try:
            del feature_extractor
        except Exception:
            pass
        model = None
        feature_extractor = None
        current_model_id = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return {"message": "모델 및 GPU 메모리 해제 완료"}
    except Exception as e:
        raise HTTPException(500, f"모델 해제 실패: {str(e)}")

@app.post("/process_nas_folder")
async def process_nas_folder(request: Request):
    """NAS 폴더의 비디오 파일들을 uploads로 복사"""
    data = await request.json()
    nas_folder = data.get("nas_folder")
    if not nas_folder:
        raise HTTPException(400, "NAS 폴더 경로가 필요합니다.")
    
    print(f"원본 경로: {nas_folder}")
    
    # 경로 보안 검증
    try:
        # NAS_BASE_PATH 부분을 제거하고 NAS_TARGET_PATH부터 시작하는 경로로 변환
        if nas_folder.startswith(NAS_BASE_PATH):
            # NAS_BASE_PATH를 NAS_TARGET_PATH로 변경
            nas_folder = nas_folder.replace(NAS_BASE_PATH, NAS_TARGET_PATH, 1)
            print(f"경로 변환: {NAS_BASE_PATH} -> {NAS_TARGET_PATH}")
        elif nas_folder.startswith('/home'):
            # /home으로 시작하는 다른 경로도 NAS_TARGET_PATH로 변경
            nas_folder = nas_folder.replace('/home', NAS_TARGET_PATH, 1)
            print(f"경로 변환: /home -> {NAS_TARGET_PATH}")
        
        # 상대 경로인 경우 현재 작업 디렉토리 기준으로 절대 경로로 변환
        if not nas_folder.startswith('/'):
            nas_path = Path.cwd() / nas_folder
            print(f"상대 경로 감지, 현재 작업 디렉토리: {Path.cwd()}")
        else:
            nas_path = Path(nas_folder)
            print(f"절대 경로 감지")
        
        nas_path = nas_path.resolve()
        print(f"정규화된 경로: {nas_path}")
        
        # 허용된 경로인지 확인 (보안상 필요한 경우)
        # NAS_TARGET_PATH로 시작하는 경로만 허용
        if not str(nas_path).startswith(NAS_TARGET_PATH):
            raise HTTPException(400, f"허용되지 않은 경로입니다. {NAS_TARGET_PATH}로 시작하는 경로만 사용 가능합니다.")
        
        # 경로가 존재하는지 확인
        if not nas_path.exists():
            raise HTTPException(400, f"경로가 존재하지 않습니다: {nas_folder}")
        
        # 경로가 디렉토리인지 확인
        if not nas_path.is_dir():
            raise HTTPException(400, f"지정된 경로가 디렉토리가 아닙니다: {nas_folder}")
            
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(400, f"경로 검증 실패: {str(e)}")
    
    # 이미 복사 중인지 확인
    if nas_copy_progress["is_copying"]:
        raise HTTPException(400, "이미 파일 복사가 진행 중입니다.")
    
    print(f"NAS 폴더 복사 시작: {nas_folder} -> {nas_path}")
    
    # 별도 스레드에서 복사 작업 실행
    loop = asyncio.get_event_loop()
    func = functools.partial(copy_nas_folder_sync, str(nas_path))
    await loop.run_in_executor(None, func)
    
    return {
        "message": "NAS 폴더 복사가 백그라운드에서 시작되었습니다.",
        "total_files": nas_copy_progress["total_files"],
        "copied_files": nas_copy_progress["copied_files"],
        "errors": nas_copy_progress["errors"]
    }

@app.get("/nas_copy_progress")
async def get_nas_copy_progress():
    """NAS 폴더 복사 진행 상태 조회"""
    return nas_copy_progress

@app.post("/cancel_nas_copy")
async def cancel_nas_copy():
    """NAS 폴더 복사 취소"""
    global nas_copy_progress
    if nas_copy_progress["is_copying"]:
        nas_copy_progress["is_copying"] = False
        return {"message": "NAS 폴더 복사가 취소되었습니다."}
    else:
        return {"message": "복사 중인 작업이 없습니다."}

@app.get("/nas_paths")
async def get_nas_paths_endpoint():
    """NAS 경로 설정 조회"""
    return {
        "base_path": NAS_BASE_PATH,
        "target_path": NAS_TARGET_PATH
    }

def push_state(state):
    while True:
        try:
            inference_state_queue.put_nowait(state)
            break
        except asyncio.QueueFull:
            try:
                inference_state_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

# === 멀티프로세스 공유 객체 및 큐 선언 ===
manager = None
inference_task_queue = None
inference_result_queue = None
overlay_task_queue = None
overlay_result_queue = None
shared_state = None

@app.on_event("startup")
def startup_event():
    global manager, inference_task_queue, inference_result_queue, overlay_task_queue, overlay_result_queue, shared_state
    import multiprocessing as mp
    manager = mp.Manager()
    inference_task_queue = manager.Queue()
    inference_result_queue = manager.Queue()
    overlay_task_queue = manager.Queue()
    overlay_result_queue = manager.Queue()
    shared_state = manager.dict()
    start_workers()

# === 추론 워커 함수 ===
def inference_worker(task_q, result_q, overlay_q, shared_state):
    import torch
    from utils import load_model, process_video
    import csv
    from pathlib import Path
    model, feature_extractor = None, None
    while True:
        task = task_q.get()
        if task is None:
            break
        # 모델 로드/캐시 (간단화)
        if model is None or feature_extractor is None or task.get('model_id') != getattr(model, 'model_id', None):
            model, feature_extractor = load_model(task['model_id'])
            if hasattr(model, 'model_id'):
                model.model_id = task['model_id']
        # 실제 추론 함수 호출
        result = process_video(
            video_path=task['video_path'],
            model=model,
            feature_extractor=feature_extractor,
            sampling_window_frames=task['interval'],
            sliding_window_step_frames=task['infer_period'],
            num_frames_to_sample=task['batch'],
            progress_callback=lambda current, total: None,  # 더미 콜백
            result_callback=lambda result: None,  # 더미 콜백
            stop_checker=lambda: False,
            frame_callback=lambda frame: None  # 더미 콜백
        )
        # CSV에 결과 저장
        if result:
            csv_path = Path("/aivanas/raw/surveillance/action/eval_results/temp_results/results.csv")
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            f_existed = csv_path.is_file()
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['video_name', 'start_time', 'end_time', 'prediction_label', 'start_frame', 'end_frame', 'inference_time_ms', 'inference_fps'])
                if not f_existed: writer.writeheader()
                writer.writerows(result)
        # 오버레이 작업을 큐에 넣기
        if result:
            overlay_task = {
                'video_path': task['video_path'],
                'results': result,
                'output_path': Path("/aivanas/raw/surveillance/action/eval_results/temp_results") / f"{task['video_path'].stem}_overlay.mp4"
            }
            overlay_q.put(overlay_task)
        # 상태 업데이트
        shared_state['processed_videos'] = shared_state.get('processed_videos', 0) + 1
        shared_state['events'].append({"type": "video_processed", "video": task['video_path'].name, "timestamp": datetime.now().isoformat()})
        
        # 모든 비디오 처리 완료 시 추론 종료
        if shared_state.get('processed_videos', 0) >= shared_state.get('total_videos', 0):
            shared_state['is_inferencing'] = False
            shared_state['events'].append({"type": "complete", "timestamp": datetime.now().isoformat()})
        
        result_q.put({'video_name': str(task['video_path']), 'results': result})

# === 오버레이 워커 함수 ===
def overlay_worker(task_q, result_q, shared_state):
    from utils import create_overlay_video
    while True:
        task = task_q.get()
        if task is None:
            break
        create_overlay_video(task['video_path'], task['results'], task['output_path'])
        shared_state['events'].append({"type": "overlay_created", "video": task['video_path'].name, "timestamp": datetime.now().isoformat()})
        result_q.put({'video_name': str(task['video_path']), 'overlay_done': True})

# === 상태 관리 프로세스 ===
def state_collector(result_queues, shared_state):
    while True:
        for q in result_queues:
            try:
                result = q.get_nowait()
                shared_state.update(result)  # 최신 상태만 반영
            except:
                pass
        import time
        time.sleep(0.1)

# === 멀티프로세스 워커 시작 ===
def start_workers():
    global inference_task_queue, inference_result_queue, overlay_task_queue, overlay_result_queue, shared_state
    
    # === 워커 풀 개수 지정 ===
    NUM_INFER_WORKERS = 4
    NUM_OVERLAY_WORKERS = 2

    # === 워커 프로세스 및 상태 관리 프로세스 시작 ===
    inf_procs = [mp.Process(target=inference_worker, args=(inference_task_queue, inference_result_queue, overlay_task_queue, shared_state)) for _ in range(NUM_INFER_WORKERS)]
    ovl_procs = [mp.Process(target=overlay_worker, args=(overlay_task_queue, overlay_result_queue, shared_state)) for _ in range(NUM_OVERLAY_WORKERS)]
    state_proc = mp.Process(target=state_collector, args=([inference_result_queue, overlay_result_queue], shared_state))
    
    for p in inf_procs + ovl_procs:
        p.start()
    state_proc.start()
    
    return inf_procs, ovl_procs, state_proc

if __name__ == "__main__":
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    # debugpy.wait_for_client()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000, log_level="critical", access_log=False)
# /backend/main.py

import logging
import sys
from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException, BackgroundTasks, Request, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from contextlib import asynccontextmanager
from datetime import datetime
import asyncio
import aiofiles
from pathlib import Path
from typing import List, Dict, Optional
import cv2
import os
import subprocess
import base64
import csv
import psutil
import shutil
import torch
import time
import multiprocessing as mp
import json
import platform

from utils import (
    load_model, process_video, create_overlay_video,
    create_results_csv, get_video_info
)
from inference_manager import InferenceManager
from websocket_manager import websocket_manager
from nas_manager import NASManager

# GPU 정보용 pynvml import 및 초기화
# pynvml 관련 import를 try-except로 감싸서 선택적으로 import
try:
    from pynvml import *
    nvmlInit()  # 반드시 import 직후에 한 번만 호출!
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    # 더미 함수들 정의
    def nvmlInit(): pass
    def nvmlShutdown(): pass
    def nvmlDeviceGetCount(): return 0
    def nvmlDeviceGetHandleByIndex(index): return None
    def nvmlDeviceGetName(handle): return "N/A"
    def nvmlDeviceGetMemoryInfo(handle):
        class DummyMem:
            used = 0.0
            total = 0.0
        return DummyMem()
    def nvmlDeviceGetUtilizationRates(handle):
        class DummyUtil:
            gpu = 0.0
        return DummyUtil()
    class NVMLError(Exception): pass

# 환경 변수 및 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = Path("/aivanas/raw/surveillance/action/eval_results/temp_results")
SAVE_RESULTS_DIR = Path("/aivanas/raw/surveillance/action/eval_results/save_results")

# NAS 경로 설정
NAS_BASE_PATH = os.getenv('NAS_BASE_PATH', '/home/hsnam')
NAS_TARGET_PATH = os.getenv('NAS_TARGET_PATH', '/aivanas')

logging.basicConfig(level=logging.CRITICAL)

# uvicorn, fastapi 등 서버 프레임워크 로그도 CRITICAL로 제한
logging.getLogger("uvicorn").setLevel(logging.CRITICAL)
logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)
logging.getLogger("fastapi").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

mp.set_start_method("spawn", force=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI 애플리케이션 시작")
    # 멀티프로세싱 객체 및 매니저를 lifespan에서만 생성
    manager = mp.Manager()
    is_inferencing = manager.Value('b', False)
    total_videos = manager.Value('i', 0)
    processed_videos = manager.Value('i', 0)
    stop_flag = manager.Value('b', False)
    video_states = manager.dict()
    video_progress = manager.dict()
    video_frames = manager.dict()
    video_results = manager.dict()
    cumulative_accuracy = manager.Value('d', 0.0)
    metrics = manager.dict({
        "tp": 0, "tn": 0, "fp": 0, "fn": 0, 
        "precision": 0.0, "recall": 0.0, "f1_score": 0.0
    })
    events = manager.list()

    app.state.manager = manager
    app.state.inference_manager = InferenceManager(
        upload_dir=UPLOAD_DIR,
        results_dir=RESULTS_DIR,
        save_results_dir=SAVE_RESULTS_DIR,
        manager=manager,
        is_inferencing=is_inferencing,
        total_videos=total_videos,
        processed_videos=processed_videos,
        stop_flag=stop_flag,
        video_states=video_states,
        video_progress=video_progress,
        video_frames=video_frames,
        video_results=video_results,
        cumulative_accuracy=cumulative_accuracy,
        metrics=metrics,
        events=events
    )
    app.state.nas_manager = NASManager(
        upload_dir=UPLOAD_DIR,
        nas_base_path=str(BASE_DIR),
        nas_target_path=str(SAVE_RESULTS_DIR)
    )
    yield
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    print("FastAPI 애플리케이션 종료")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ===================== 추론 관련 =====================
@app.post("/infer")
async def start_inference_endpoint(request: Request):
    data = await request.json()
    interval = data.get("interval", 90)
    infer_period = data.get("infer_period", 30)
    batch = data.get("batch", 16)
    annotation_data = data.get("annotation_data", {})
    model_id = data.get("model_id")
    if not model_id:
        raise HTTPException(400, "model_id가 필요합니다.")
    video_files = sorted([p for p in UPLOAD_DIR.glob('*') if p.suffix in ['.mp4', '.avi', '.mov']])
    if not video_files:
        raise HTTPException(400, "업로드된 비디오가 없습니다.")
    inference_manager = request.app.state.inference_manager
    inference_manager.start_inference(video_files, interval, infer_period, batch, model_id, annotation_data)
    return {"message": "추론이 백그라운드에서 시작되었습니다."}

@app.post("/stop_infer")
async def stop_infer_endpoint(request: Request):
    inference_manager = request.app.state.inference_manager
    inference_manager.stop_inference()
    return {"message": "추론 중지 및 결과 이동 완료"}

@app.get("/current_inference_state")
async def get_current_inference_state_endpoint(request: Request):
    inference_manager = request.app.state.inference_manager
    return inference_manager.get_state()

# ===================== 모델 관련 =====================
@app.post("/model")
async def set_model_endpoint(request: Request):
    data = await request.json()
    model_id = data.get("model_id")
    if not model_id:
        raise HTTPException(400, "model_id가 필요합니다.")
    inference_manager = request.app.state.inference_manager
    inference_manager.load_model(model_id)
    return {"message": "모델 로드 완료"}

@app.get("/current_model")
async def get_current_model_endpoint(request: Request):
    inference_manager = request.app.state.inference_manager
    return {"model_id": inference_manager.current_model_id}

@app.post("/unload_model")
async def unload_model_endpoint(request: Request):
    inference_manager = request.app.state.inference_manager
    inference_manager.cleanup()
    return {"message": "모델 및 리소스 해제 완료"}

# ===================== 업로드/파일 관리 =====================
@app.post("/upload")
async def upload_videos_endpoint(request: Request):
    uploaded = []
    skipped = []
    form = await request.form()
    files = form.getlist("files")
    paths = form.getlist("paths")
    for i, file in enumerate(files):
        relative_path = Path(paths[i])
        full_path = UPLOAD_DIR / Path(relative_path).name
        if full_path.exists():
            skipped.append(str(relative_path))
            continue
        async with aiofiles.open(full_path, 'wb') as f:
            await f.write(await file.read())
        uploaded.append({"name": str(relative_path), "size": full_path.stat().st_size})
    return {"files": uploaded, "skipped": skipped}

@app.get("/uploads")
async def get_uploads_endpoint():
    files = []
    for root, _, filenames in os.walk(UPLOAD_DIR):
        for filename in filenames:
            full_path = Path(root) / filename
            files.append({
                "name": str(full_path.relative_to(UPLOAD_DIR)),
                "size": full_path.stat().st_size
            })
    return {"files": files}

@app.delete("/upload/{filename}")
async def delete_file_endpoint(filename: str):
    path = UPLOAD_DIR / filename
    if path.exists():
        path.unlink()
    return {"message": f"{filename} 삭제 완료"}

@app.delete("/uploads")
async def delete_all_files_endpoint(request: Request):
    for f in UPLOAD_DIR.glob("*"):
        if f.is_file():
            f.unlink()
    inference_manager = request.app.state.inference_manager
    inference_manager.cleanup()
    return {"message": "모든 업로드 파일 삭제 완료"}

# ===================== NAS 폴더 관련 =====================
@app.post("/process_nas_folder")
async def process_nas_folder(request: Request):
    data = await request.json()
    nas_folder = data.get("nas_folder")
    if not nas_folder:
        raise HTTPException(400, "NAS 폴더 경로가 필요합니다.")
    nas_manager = request.app.state.nas_manager
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, nas_manager.copy_nas_folder, nas_folder)
    return nas_manager.get_progress()

@app.get("/nas_copy_progress")
async def get_nas_copy_progress(request: Request):
    """NAS 폴더 복사 진행 상태 조회"""
    nas_manager = request.app.state.nas_manager
    return nas_manager.get_progress()

@app.post("/cancel_nas_copy")
async def cancel_nas_copy(request: Request):
    """NAS 폴더 복사 취소"""
    nas_manager = request.app.state.nas_manager
    return nas_manager.cancel_copy()

@app.get("/nas_paths")
async def get_nas_paths_endpoint():
    """NAS 경로 설정 조회"""
    return {
        "base_path": NAS_BASE_PATH,
        "target_path": NAS_TARGET_PATH
    }

# ===================== 웹소켓 =====================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            inference_manager = websocket.app.state.inference_manager
            await websocket_manager.broadcast_state(inference_manager.get_state())
            await asyncio.sleep(0.2)
    except Exception:
        websocket_manager.disconnect(websocket)

@app.websocket("/ws/realtime_overlay")
async def websocket_realtime_overlay_endpoint(websocket: WebSocket):
    await websocket.accept()  # 연결 수립
    video_name = (await websocket.receive_text()).strip()
    websocket_manager.set_overlay_connection(video_name, websocket)
    try:
        while True:
            inference_manager = websocket.app.state.inference_manager
            frame = inference_manager.get_video_frame(video_name)
            if frame:
                await websocket_manager.broadcast_frame(video_name, frame)
            await asyncio.sleep(0.05)
    except Exception:
        websocket_manager.disconnect(websocket)

# ===================== 결과/메타데이터 =====================
@app.get("/results/videos")
async def get_result_videos_endpoint():
    return {"videos": sorted([f.name for f in RESULTS_DIR.glob("*_overlay.mp4")])}

@app.get("/video/{video_id}/overlay")
async def get_overlay_video_endpoint(video_id: str):
    path = RESULTS_DIR / f"{Path(video_id).stem}_overlay.mp4"
    if not path.exists():
        raise HTTPException(404, "파일 없음")
    file_size = os.path.getsize(path)
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=path.name,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size)
        }
    )

@app.get("/results.csv")
async def get_results_csv_endpoint():
    path = RESULTS_DIR / "results.csv"
    if not path.exists():
        raise HTTPException(404, "파일 없음")
    return FileResponse(path)

@app.get("/video_metadata/{video_path}")
async def get_video_metadata_endpoint(video_path: str):
    try:
        metadata = get_video_info(Path(video_path))
        if metadata:
            return metadata
        else:
            raise HTTPException(500, "메타데이터 가져오기 실패")
    except Exception as e:
        print(f"메타데이터 가져오기 실패: {str(e)}")
        raise HTTPException(500, "메타데이터 가져오기 실패")

@app.post("/pause_infer")
async def pause_infer_endpoint(request: Request):
    inference_manager = request.app.state.inference_manager
    inference_manager.STOP_FLAG.value = True
    return {"message": "추론 일시정지"}

@app.post("/resume_infer")
async def resume_infer_endpoint(request: Request):
    inference_manager = request.app.state.inference_manager
    inference_manager.STOP_FLAG.value = False
    return {"message": "추론 재개"}

@app.get("/system_info")
async def get_system_info_endpoint():
    info = {"cpu": None, "ram": None, "ram_used_mb": None, "ram_total_mb": None, "gpu": None, "gpuMem": None, "gpuUtil": None}
    try:
        info["cpu"] = psutil.cpu_percent(0.1)
        vmem = psutil.virtual_memory()
        info["ram"], info["ram_used_mb"], info["ram_total_mb"] = vmem.percent, vmem.used / 1e6, vmem.total / 1e6
    except Exception as e:
        print(f"CPU/RAM info error: {e}")
    try:
        if NVML_AVAILABLE and nvmlDeviceGetCount() > 0:
            handle = nvmlDeviceGetHandleByIndex(0)
            gpu_name = nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode()
            info["gpu"] = gpu_name
            mem = nvmlDeviceGetMemoryInfo(handle)
            info["gpuMem"] = f"{mem.used/1e6:.0f}MB / {mem.total/1e6:.0f}MB"
            info["gpuUtil"] = float(nvmlDeviceGetUtilizationRates(handle).gpu)
            print(f"GPU: {info['gpu']}, GPU MEM: {info['gpuMem']}, GPU UTIL: {info['gpuUtil']}")
        else:
            print("NVML not available or no GPU found.")
    except NVMLError as e:
        print(f"NVML error: {e}")
    except Exception as e:
        print(f"GPU info error: {e}")
    return info

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    import traceback
    tb = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print(f"[EXCEPTION] {tb}")
    return PlainTextResponse(tb, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
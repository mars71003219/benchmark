# /backend/main.py

from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from datetime import datetime
import asyncio
import aiofiles
from pathlib import Path
from typing import List, Dict
import functools
import cv2
import os
import subprocess
import sys

import psutil
from pynvml import *

from utils import (
    load_model, process_video, create_overlay_video,
    create_results_csv, get_video_info
)

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

try:
    nvmlInit()
except NVMLError: pass

@app.on_event("shutdown")
def shutdown_event():
    try: nvmlShutdown()
    except NVMLError: pass

def reset_inference_state():
    global current_video_info, stop_infer_flag
    stop_infer_flag = False
    current_video_info = {
        "total_videos": 0, "processed_videos": 0, "current_video": None,
        "current_progress": 0, "events": [], "per_video_progress": {},
        "is_inferencing": False,
    }

def stop_checker():
    return stop_infer_flag

# [수정] 이 함수는 이제 동기적으로 실행됨 (run_in_executor에서 호출되므로)
def process_all_videos_sync(interval, infer_period, batch, save_dir):
    reset_inference_state()
    current_video_info["is_inferencing"] = True
    video_files = sorted([p for p in UPLOAD_DIR.glob('*') if p.suffix in ['.mp4', '.avi', '.mov']])
    current_video_info["total_videos"] = len(video_files)
    
    all_results = []
    for i, video_path in enumerate(video_files):
        if stop_checker():
            current_video_info["events"].append({"type": "stop", "timestamp": datetime.now().isoformat()})
            break

        current_video_info["current_video"] = video_path.name
        current_video_info["events"].append({"type": "start", "video": video_path.name, "timestamp": datetime.now().isoformat()})

        def progress_callback(done, total):
            current_video_info["current_progress"] = int(done / total * 100) if total > 0 else 0

        def result_callback(result: Dict):
            current_video_info["events"].append({
                "type": "detection", "video": video_path.name,
                "timestamp": datetime.now().isoformat(), "data": result
            })

        try:
            video_results = process_video(
                video_path=video_path, model=model, feature_extractor=feature_extractor,
                sampling_window_frames=interval, sliding_window_step_frames=infer_period,
                num_frames_to_sample=batch, progress_callback=progress_callback,
                result_callback=result_callback, stop_checker=stop_checker
            )
            all_results.extend(video_results)
            if video_results:
                overlay_path = save_dir / f"{video_path.stem}_overlay.mp4"
                create_overlay_video(video_path, video_results, overlay_path)
            
            current_video_info["events"].append({"type": "video_processed", "video": video_path.name, "timestamp": datetime.now().isoformat()})
            current_video_info["processed_videos"] = i + 1
        except Exception as e:
            print(f"비디오 처리 중 에러 발생: {e}")
            current_video_info["events"].append({"type": "error", "message": str(e), "timestamp": datetime.now().isoformat()})
            current_video_info["processed_videos"] = i + 1

    if all_results:
        create_results_csv(all_results, save_dir / "results.csv")
    current_video_info["events"].append({"type": "complete", "timestamp": datetime.now().isoformat()})
    current_video_info["is_inferencing"] = False


@app.post("/infer")
async def start_inference_endpoint(request: Request):
    if not model: raise HTTPException(400, "모델이 로드되지 않았습니다.")
    if not any(UPLOAD_DIR.iterdir()): raise HTTPException(400, "업로드된 비디오가 없습니다.")
    if current_video_info.get("is_inferencing"): raise HTTPException(400, "이미 추론이 진행 중입니다.")
    
    data = await request.json()
    interval = data.get("interval", 90)
    infer_period = data.get("infer_period", 30)
    batch = data.get("batch", 16)
    
    # [핵심 수정] 이벤트 루프를 막지 않기 위해 별도 스레드에서 동기 함수 실행
    loop = asyncio.get_event_loop()
    # functools.partial을 사용해 함수에 인자 전달
    func = functools.partial(process_all_videos_sync, interval, infer_period, batch, RESULTS_DIR)
    # 별도 스레드에서 작업 실행
    await loop.run_in_executor(None, func)
    
    return {"message": "추론이 백그라운드에서 시작되었습니다."}

# ... (다른 API 엔드포인트들은 이전과 동일) ...
@app.post("/model")
async def set_model_endpoint(request: Request):
    global model, feature_extractor
    data = await request.json()
    model_id = data.get("model_id")
    if not model_id: raise HTTPException(400, "model_id가 필요합니다.")
    model, feature_extractor = load_model(model_id)
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
    return {"files": sorted(files, key=lambda x: x['name'])}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(current_video_info)
            await asyncio.sleep(0.2)
    except Exception: pass

@app.post("/stop_infer")
async def stop_infer_endpoint():
    global stop_infer_flag
    stop_infer_flag = True
    return {"message": "추론 중지 요청"}

@app.delete("/upload/{filename}")
async def delete_file_endpoint(filename: str):
    path = UPLOAD_DIR / filename
    if path.exists(): path.unlink()
    return {"message": f"{filename} 삭제 완료"}

@app.delete("/uploads")
async def delete_all_files_endpoint():
    for f in UPLOAD_DIR.glob("*"):
        if f.is_file(): f.unlink()
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
    except Exception: pass
    try:
        if nvmlDeviceGetCount() > 0:
            handle = nvmlDeviceGetHandleByIndex(0)
            info["gpu"] = nvmlDeviceGetName(handle)
            mem = nvmlDeviceGetMemoryInfo(handle)
            info["gpuMem"] = f"{mem.used/1e6:.0f}MB / {mem.total/1e6:.0f}MB"
            info["gpuUtil"] = nvmlDeviceGetUtilizationRates(handle).gpu
    except NVMLError: pass
    return info

@app.get("/results/videos")
async def get_result_videos_endpoint():
    return {"videos": sorted([f.name for f in RESULTS_DIR.glob("*_overlay.mp4")])}

@app.get("/video/{video_id}/overlay")
async def get_overlay_video_endpoint(video_id: str):
    path = RESULTS_DIR / f"{Path(video_id).stem}_overlay.mp4"
    if not path.exists(): raise HTTPException(404, "파일 없음")
    return FileResponse(path)

@app.get("/results.csv")
async def get_results_csv_endpoint():
    path = RESULTS_DIR / "results.csv"
    if not path.exists(): raise HTTPException(404, "파일 없음")
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

if __name__ == "__main__":
    import uvicorn
    reset_inference_state()
    uvicorn.run(app, host="0.0.0.0", port=10000)


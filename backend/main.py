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
import csv
import psutil
from pynvml import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

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
RESULTS_DIR = Path("/aivanas/raw/action/eval_results/temp_results")
SAVE_RESULTS_DIR = Path("/aivanas/raw/action/eval_results/save_results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SAVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

pause_infer_flag = False
resume_infer_flag = False
paused_video_name = None

try:
    nvmlInit()
except NVMLError: pass

@app.on_event("shutdown")
def shutdown_event():
    try: nvmlShutdown()
    except NVMLError: pass

def reset_inference_state():
    global current_video_info, stop_infer_flag, global_tp, global_tn, global_fp, global_fn, global_total_processed_clips, global_correct_predictions
    stop_infer_flag = False
    current_video_info = {
        "total_videos": 0, "processed_videos": 0, "current_video": None,
        "current_progress": 0, "events": [], "per_video_progress": {},
        "is_inferencing": False,
        "cumulative_accuracy": 0.0, # New field
        "metrics": {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}, # New field
    }
    # Reset global counters too
    global_tp = 0
    global_tn = 0
    global_fp = 0
    global_fn = 0
    global_total_processed_clips = 0
    global_correct_predictions = 0
    # Clear queues when state is reset for new inference
    while not realtime_frame_queue.empty():
        try: realtime_frame_queue.get_nowait()
        except asyncio.QueueEmpty: pass
    while not inference_state_queue.empty():
        try: inference_state_queue.get_nowait()
        except asyncio.QueueEmpty: pass
    # Also push initial state to queue
    try: inference_state_queue.put_nowait(current_video_info.copy())
    except asyncio.QueueFull: pass

def stop_checker():
    return stop_infer_flag

def remove_video_results_from_csv(csv_path, video_name):
    if not csv_path.exists():
        return
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df['video_name'] != video_name]
    df.to_csv(csv_path, index=False)

def process_all_videos_sync(interval, infer_period, batch, save_dir, inference_mode: str, annotation_data: Dict):
    global global_tp, global_tn, global_fp, global_fn, global_total_processed_clips, global_correct_predictions
    global pause_infer_flag, resume_infer_flag, paused_video_name
    reset_inference_state() # This will also reset global counters
    current_video_info["is_inferencing"] = True # Set to true initially
    try:
        # Send initial "is_inferencing: True" state
        try: inference_state_queue.put_nowait(current_video_info.copy())
        except asyncio.QueueFull: pass

        video_files = sorted([p for p in UPLOAD_DIR.glob('*') if p.suffix in ['.mp4', '.avi', '.mov']])
        current_video_info["total_videos"] = len(video_files)
        try: inference_state_queue.put_nowait(current_video_info.copy())
        except asyncio.QueueFull: pass
        
        final_results = []  # 최종 결과를 저장할 리스트
        overlay_tasks = []  # 오버레이 비디오 생성 작업을 저장할 리스트

        resume_from_video = None
        if resume_infer_flag and paused_video_name:
            resume_from_video = paused_video_name
            resume_infer_flag = False
        for i, video_path in enumerate(video_files):
            # RESUME: 중단된 비디오부터 시작
            if resume_from_video:
                if video_path.name != resume_from_video:
                    continue
                else:
                    resume_from_video = None
                    # results.csv에서 해당 비디오 결과 삭제
                    csv_path = save_dir / "results.csv"
                    remove_video_results_from_csv(csv_path, video_path.name)
            # PAUSE 체크
            while pause_infer_flag:
                paused_video_name = video_path.name
                import time
                time.sleep(0.5)
            if stop_checker():
                current_video_info["events"].append({"type": "stop", "timestamp": datetime.now().isoformat()})
                break

            current_video_info["current_video"] = video_path.name
            current_video_info["events"].append({"type": "start", "video": video_path.name, "timestamp": datetime.now().isoformat()})
            try: inference_state_queue.put_nowait(current_video_info.copy())
            except asyncio.QueueFull: pass

            # Get annotations for the current video
            current_video_name = video_path.name
            video_annotations_for_current_video = annotation_data.get(current_video_name, {})
            video_results = []  # 현재 비디오의 모든 추론 결과를 저장

            def progress_callback(done, total):
                current_video_info["current_progress"] = int(done / total * 100) if total > 0 else 0
                try: inference_state_queue.put_nowait(current_video_info.copy())
                except asyncio.QueueFull: pass

            def result_callback(result: Dict):
                # UI 업데이트를 위한 이벤트만 전송
                current_video_info["events"].append({
                    "type": "detection", "video": video_path.name,
                    "timestamp": datetime.now().isoformat(), "data": result
                })
                try: inference_state_queue.put_nowait(current_video_info.copy())
                except asyncio.QueueFull: pass
                
                # 현재 비디오의 결과 저장
                video_results.append(result)

            def frame_callback(frame_data: bytes):
                try:
                    realtime_frame_queue.put_nowait(frame_data)
                except asyncio.QueueFull:
                    pass # Drop frame if consumer is not ready

            def pause_or_stop_checker():
                global pause_infer_flag, paused_video_name, stop_infer_flag, current_video_info
                while pause_infer_flag:
                    paused_video_name = current_video_info.get("current_video")
                    import time
                    time.sleep(0.5)
                return stop_infer_flag

            try:
                process_video(
                    video_path=video_path, model=model, feature_extractor=feature_extractor,
                    sampling_window_frames=interval, sliding_window_step_frames=infer_period,
                    num_frames_to_sample=batch, progress_callback=progress_callback,
                    result_callback=result_callback, stop_checker=pause_or_stop_checker,
                    frame_callback=frame_callback,
                )

                # 비디오 클립 단위 결과 계산
                if video_results:
                    csv_path = save_dir / "results.csv"
                    f_existed = csv_path.is_file()
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=['video_name', 'start_time', 'end_time', 'prediction_label', 'start_frame', 'end_frame', 'inference_time_ms', 'inference_fps'])
                        if not f_existed:
                            writer.writeheader()
                        writer.writerows(video_results)

                    # 모든 라벨이 Non으로 시작하는지 확인
                    all_labels = [r["prediction_label"] for r in video_results]
                    all_are_non = all(label.lower().startswith("non") for label in all_labels)
                    
                    final_label_for_video = None
                    if all_are_non:
                        # 모든 라벨이 Non으로 시작하면 첫 번째 Non 라벨을 선택
                        final_label_for_video = all_labels[0]
                    else:
                        # Non이 아닌 라벨이 하나라도 있으면, 가장 많이 나오는 이벤트 라벨을 선택
                        from collections import Counter
                        event_labels = [label for label in all_labels if not label.lower().startswith("non")]
                        if event_labels:
                            final_label_for_video = Counter(event_labels).most_common(1)[0][0]
                    
                    if final_label_for_video:
                        # 최종 결과 저장
                        anno_label = None
                        if "AR" in video_annotations_for_current_video:
                            anno_label = video_annotations_for_current_video["AR"].get("label")
                        # metrics 계산
                        metrics_val = None
                        if anno_label:
                            if anno_label == "Fight":
                                if final_label_for_video == "Fight": metrics_val = "TP"
                                else: metrics_val = "FP"
                            elif anno_label == "NonFight":
                                if final_label_for_video == "NonFight": metrics_val = "TN"
                                else: metrics_val = "FN"
                        final_result = {
                            "video_name": video_path.name,
                            "final_pred_label": final_label_for_video,
                            "anno_label": anno_label,
                            "metrics": metrics_val
                        }
                        final_results.append(final_result)
                        
                        # Ground truth와 비교하여 메트릭 업데이트
                        if "AR" in video_annotations_for_current_video:
                            gt_label = video_annotations_for_current_video["AR"].get("label")
                            if gt_label:
                                if gt_label == "Fight":  # Assuming 'Fight' is the positive class
                                    if final_label_for_video == "Fight": global_tp += 1
                                    else: global_fp += 1
                                elif gt_label == "NonFight":
                                    if final_label_for_video == "NonFight": global_tn += 1
                                    else: global_fn += 1
                                
                                global_total_processed_clips += 1
                                if final_label_for_video == gt_label:
                                    global_correct_predictions += 1
                                
                                # Update cumulative accuracy and metrics
                                if global_total_processed_clips > 0:
                                    current_video_info["cumulative_accuracy"] = global_correct_predictions / global_total_processed_clips
                                
                                precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
                                recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
                                f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                                current_video_info["metrics"] = {
                                    "tp": global_tp, "tn": global_tn, "fp": global_fp, "fn": global_fn,
                                    "precision": round(precision, 2), "recall": round(recall, 2), "f1_score": round(f1_score, 2)
                                }
                                try: inference_state_queue.put_nowait(current_video_info.copy())
                                except asyncio.QueueFull: pass

                    # 오버레이 비디오 생성 작업 추가
                    if video_results:
                        overlay_path = save_dir / f"{video_path.stem}_overlay.mp4"
                        overlay_tasks.append((video_path, video_results, overlay_path))
                
                current_video_info["events"].append({"type": "video_processed", "video": video_path.name, "timestamp": datetime.now().isoformat()})
                current_video_info["processed_videos"] = i + 1
                try: inference_state_queue.put_nowait(current_video_info.copy())
                except asyncio.QueueFull: pass
            except Exception as e:
                print(f"비디오 처리 중 에러 발생: {e}")
                current_video_info["events"].append({"type": "error", "message": str(e), "timestamp": datetime.now().isoformat()})
                current_video_info["processed_videos"] = i + 1
                try: inference_state_queue.put_nowait(current_video_info.copy())
                except asyncio.QueueFull: pass
        
            # 최종 결과를 CSV 파일로 저장
            if final_result:
                csv_path = save_dir / "final_results.csv"
                field_names = ['video_name', 'final_pred_label', 'anno_label', 'metrics']

                # 1. 파일 존재 여부를 미리 확인한다.
                file_exists = csv_path.is_file()

                # 2. 'a'(append) 모드로 파일을 연다.
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=field_names)

                    # 3. 파일이 없을 때만 헤더를 쓴다.
                    if not file_exists:
                        writer.writeheader()

                    # 4. writerow() 메서드로 단일 딕셔너리를 추가한다.
                    writer.writerow(final_result)

            # 멀티스레드로 오버레이 비디오 생성
            if overlay_tasks:
                with ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(overlay_tasks))) as executor:
                    futures = []
                    for video_path, results, output_path in overlay_tasks:
                        future = executor.submit(create_overlay_video, video_path, results, output_path)
                        futures.append(future)
                    
                    # 모든 오버레이 비디오 생성 완료 대기
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"오버레이 비디오 생성 중 에러 발생: {e}")
        
        current_video_info["events"].append({"type": "complete", "timestamp": datetime.now().isoformat()})
        # 모든 비디오 추론이 정상적으로 끝났을 때 결과 파일 이동 및 csv 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        done_dir = SAVE_RESULTS_DIR / f'{timestamp}_done_success'
        done_dir.mkdir(parents=True, exist_ok=True)
        for f in RESULTS_DIR.glob("*"):
            shutil.move(str(f), str(done_dir / f.name))
        # 최종 결과 csv 저장
        if final_results:
            import pandas as pd
            df = pd.DataFrame(final_results)
            df.to_csv(done_dir / 'final_results.csv', index=False)
        # 전체 metrics csv 저장
        metrics_row = current_video_info.get("metrics", {})
        if metrics_row:
            import pandas as pd
            df_metrics = pd.DataFrame([metrics_row])
            df_metrics.to_csv(done_dir / 'model_evaluation_metrics.csv', index=False)
    finally:
        current_video_info["is_inferencing"] = False # Ensure this is always set to False
        try: inference_state_queue.put_nowait(current_video_info.copy()) # Final state update
        except asyncio.QueueFull: pass


@app.post("/infer")
async def start_inference_endpoint(request: Request):
    if not model: raise HTTPException(400, "모델이 로드되지 않았습니다.")
    if not any(UPLOAD_DIR.iterdir()): raise HTTPException(400, "업로드된 비디오가 없습니다.")
    if current_video_info.get("is_inferencing"): raise HTTPException(400, "이미 추론이 진행 중입니다.")
    
    data = await request.json()
    interval = data.get("interval", 90)
    infer_period = data.get("infer_period", 30)
    batch = data.get("batch", 16)
    inference_mode = data.get("inference_mode", "default") # 'AR' or 'AL'
    annotation_data = data.get("annotation_data", {}) # Annotation data from frontend
    
    # [핵심 수정] 이벤트 루프를 막지 않기 위해 별도 스레드에서 동기 함수 실행
    loop = asyncio.get_event_loop()
    # functools.partial을 사용해 함수에 인자 전달
    func = functools.partial(process_all_videos_sync, interval, infer_period, batch, RESULTS_DIR, inference_mode, annotation_data)
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
        # Send current state immediately upon connection
        await websocket.send_json(current_video_info.copy()) 

        while True:
            # Wait for data from the inference_state_queue
            state = await inference_state_queue.get()
            await websocket.send_json(state)
            inference_state_queue.task_done()
            
            # Send ping to keep connection alive
            await websocket.send_text("ping")
            try:
                pong = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                if pong != "pong":
                    raise WebSocketDisconnect()
            except asyncio.TimeoutError:
                raise WebSocketDisconnect()
    except WebSocketDisconnect:
        print("Main WebSocket disconnected by client.")
    except Exception as e:
        print(f"Main WebSocket error: {e}")
    finally:
        print("Main WebSocket connection closed.")

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

@app.websocket("/ws/realtime_overlay")
async def websocket_realtime_overlay_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            frame_data = await realtime_frame_queue.get() # Blocking get until a frame is available
            encoded_frame = base64.b64encode(frame_data).decode('utf-8')
            await websocket.send_text(encoded_frame)
            
            # Send ping to keep connection alive
            await websocket.send_text("ping")
            try:
                pong = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                if pong != "pong":
                    raise WebSocketDisconnect()
            except asyncio.TimeoutError:
                raise WebSocketDisconnect()
    except WebSocketDisconnect:
        print("실시간 오버레이 WebSocket 연결이 클라이언트에 의해 종료되었습니다.")
    except Exception as e:
        print(f"실시간 오버레이 WebSocket 오류: {e}")
    finally:
        print("실시간 오버레이 WebSocket 연결 종료.")

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

if __name__ == "__main__":
    import uvicorn
    reset_inference_state()
    uvicorn.run(app, host="0.0.0.0", port=10000)


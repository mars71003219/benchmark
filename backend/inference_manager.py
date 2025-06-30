# /backend/inference_manager.py

import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import torch
import gc
import os
from transformers import AutoModelForVideoClassification, AutoFeatureExtractor
from huggingface_hub import snapshot_download
from multiprocessing import Queue, Process
import queue
import threading

# 멀티프로세싱 시작 방식 설정 (spawn 방식으로 설정)
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 이미 설정된 경우 무시

from utils import load_model, process_video, create_overlay_video, create_results_csv

def inference_worker_initialized(worker_id: int, model_cache_path: str, video_queue, result_queue, stop_event, 
                                shared_video_progress, shared_video_results, shared_video_frames, shared_events, 
                                shared_metrics, annotation_data=None):
    """초기화된 추론 워커 프로세스 - 모델을 한 번만 로딩하고 비디오를 큐에서 가져와 처리"""
    from pathlib import Path
    import torch
    from transformers import AutoModelForVideoClassification, AutoFeatureExtractor
    from datetime import datetime
    import random
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"워커 {worker_id}: 모델 로딩 시작")
    # 모델을 한 번만 로딩
    model = AutoModelForVideoClassification.from_pretrained(model_cache_path, local_files_only=True).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_cache_path, local_files_only=True)
    model.eval()
    print(f"워커 {worker_id}: 모델 로딩 완료")
    
    # 랜덤 스트림 비디오 선택 (워커별로 다른 비디오 선택)
    stream_video_name = None
    
    while not stop_event.is_set():
        try:
            # 비디오 작업을 큐에서 가져오기 (5초 타임아웃)
            video_task = video_queue.get(timeout=5)
            
            if video_task is None:  # 종료 신호
                break
                
            video_path_str, interval, infer_period, batch, task_annotation_data, results_dir_str = video_task
            video_path = Path(video_path_str)
            results_dir = Path(results_dir_str)
            
            print(f"워커 {worker_id}: 비디오 처리 시작 - {video_path.name}")
            
            # 랜덤 스트림 비디오 설정 (처음 처리하는 비디오를 스트림용으로 선택)
            if stream_video_name is None:
                stream_video_name = video_path.name
            
            # 비디오 처리 (이미 로딩된 모델 사용)
            try:
                from utils import process_video, create_overlay_video, create_results_csv
                
                # 실시간 진행률 및 결과 처리를 위한 콜백 함수들
                def progress_callback(current, total):
                    """실시간 진행률 업데이트"""
                    progress_percent = (current / total) * 100 if total > 0 else 0
                    shared_video_progress[video_path.name] = progress_percent
                    
                    # 실시간 이벤트 기록
                    shared_events.append({
                        "type": "progress_update",
                        "video_name": video_path.name,
                        "current": current,
                        "total": total,
                        "progress": progress_percent,
                        "timestamp": datetime.now().isoformat()
                    })
                
                def result_callback(result):
                    """개별 추론 결과 즉시 처리"""
                    # 결과를 즉시 저장
                    shared_video_results[video_path.name] = result
                    
                    # 실시간 이벤트 기록
                    shared_events.append({
                        "type": "inference_result",
                        "video_name": video_path.name,
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # 메트릭 업데이트 (annotation_data가 있는 경우)
                    if task_annotation_data and video_path.name in task_annotation_data:
                        _update_metrics_worker(video_path.name, [result], task_annotation_data[video_path.name], shared_metrics)
                
                def stop_checker():
                    """중지 신호 확인"""
                    return stop_event.is_set()
                
                def frame_callback(frame_data):
                    """실시간 프레임 스트림"""
                    # 랜덤하게 선택된 비디오의 프레임만 저장 (실시간 스트림용)
                    if video_path.name == stream_video_name:
                        shared_video_frames[video_path.name] = frame_data
                        
                        # 실시간 이벤트 기록
                        shared_events.append({
                            "type": "frame_update",
                            "video_name": video_path.name,
                            "timestamp": datetime.now().isoformat()
                        })
                
                # 비디오 처리 실행
                results = process_video(
                    video_path=video_path,
                    model=model,
                    feature_extractor=feature_extractor,
                    sampling_window_frames=interval,
                    sliding_window_step_frames=infer_period,
                    num_frames_to_sample=batch,
                    progress_callback=progress_callback,
                    result_callback=result_callback,
                    stop_checker=stop_checker,
                    frame_callback=frame_callback
                )
                
                # 결과 저장
                if results:
                    # 오버레이 비디오 생성
                    overlay_path = results_dir / f"{video_path.stem}_overlay.mp4"
                    create_overlay_video(video_path, results, overlay_path)
                    
                    # CSV 결과 저장
                    csv_path = results_dir / f"{video_path.stem}_results.csv"
                    create_results_csv(results, csv_path)
                
                # 결과를 결과 큐에 전송
                result = {
                    "worker_id": worker_id,
                    "video_name": video_path.name,
                    "status": "completed",
                    "results_count": len(results) if results else 0,
                    "overlay_path": str(overlay_path) if results else None,
                    "csv_path": str(csv_path) if results else None,
                    "results": results  # 전체 결과 데이터 포함
                }
                result_queue.put(result)
                
                print(f"워커 {worker_id}: 비디오 처리 완료 - {video_path.name} (결과: {len(results) if results else 0}개)")
                
            except Exception as e:
                print(f"워커 {worker_id}: 비디오 처리 중 오류 발생 - {video_path.name}: {e}")
                result_queue.put({
                    "worker_id": worker_id,
                    "video_name": video_path.name,
                    "error": str(e),
                    "status": "error"
                })
            
        except queue.Empty:
            # 큐가 비어있으면 계속 대기
            continue
        except Exception as e:
            print(f"워커 {worker_id}: 오류 발생 - {e}")
            result_queue.put({
                "worker_id": worker_id,
                "error": str(e),
                "status": "error"
            })
    
    # 메모리 정리
    del model
    del feature_extractor
    torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
    print(f"워커 {worker_id}: 종료")

def _update_metrics_worker(video_name: str, results: list, annotation: dict, metrics_dict: Dict):
    """워커 프로세스에서 사용하는 메트릭 업데이트 함수"""
    tp = tn = fp = fn = 0
    gt_label = annotation.get('label') if isinstance(annotation, dict) else None
    if gt_label is None:
        return
    for result in results:
        pred = result['prediction_label']
        if pred == 'Fight' and gt_label == 'Fight':
            tp += 1
        elif pred == 'NonFight' and gt_label == 'NonFight':
            tn += 1
        elif pred == 'Fight' and gt_label == 'NonFight':
            fp += 1
        elif pred == 'NonFight' and gt_label == 'Fight':
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    metrics_dict['tp'] = metrics_dict.get('tp', 0) + tp
    metrics_dict['tn'] = metrics_dict.get('tn', 0) + tn
    metrics_dict['fp'] = metrics_dict.get('fp', 0) + fp
    metrics_dict['fn'] = metrics_dict.get('fn', 0) + fn
    metrics_dict['precision'] = precision
    metrics_dict['recall'] = recall
    metrics_dict['f1_score'] = f1

class InferenceManager:
    """멀티프로세싱 기반 추론 관리자 (프로세스 풀 + 큐 방식)"""
    def __init__(self, upload_dir: Path, results_dir: Path, save_results_dir: Path,
                 manager=None,
                 is_inferencing=None, total_videos=None, processed_videos=None, stop_flag=None,
                 video_states=None, video_progress=None, video_frames=None, video_results=None,
                 cumulative_accuracy=None, metrics=None, events=None):
        self.upload_dir = upload_dir
        self.results_dir = results_dir
        self.save_results_dir = save_results_dir
        self.current_model_id = None
        self.current_model_cache_path = None
        
        # 프로세스 풀 관련
        self.worker_processes = []
        self.video_queue = None
        self.result_queue = None
        self.stop_event = None
        self.max_workers = 4  # 최대 워커 수
        
        # 멀티프로세싱 공유 객체
        self.manager = manager or mp.Manager()
        self.IS_INFERENCING = is_inferencing or self.manager.Value('b', False)
        self.TOTAL_VIDEOS = total_videos or self.manager.Value('i', 0)
        self.PROCESSED_VIDEOS = processed_videos or self.manager.Value('i', 0)
        self.STOP_FLAG = stop_flag or self.manager.Value('b', False)
        self.VIDEO_STATES = video_states or self.manager.dict()
        self.VIDEO_PROGRESS = video_progress or self.manager.dict()
        self.VIDEO_FRAMES = video_frames or self.manager.dict()
        self.VIDEO_RESULTS = video_results or self.manager.dict()
        self.CUMULATIVE_ACCURACY = cumulative_accuracy or self.manager.Value('d', 0.0)
        self.METRICS = metrics or self.manager.dict({
            "tp": 0, "tn": 0, "fp": 0, "fn": 0, 
            "precision": 0.0, "recall": 0.0, "f1_score": 0.0
        })
        self.EVENTS = events or self.manager.list()
    
    def _create_worker_pool(self, model_cache_path: str, annotation_data: Optional[Dict] = None):
        """워커 프로세스 풀 생성 및 모델 로딩"""
        if self.worker_processes:
            self._cleanup_worker_pool()
        
        # 큐와 이벤트 생성
        self.video_queue = Queue()
        self.result_queue = Queue()
        self.stop_event = mp.Event()
        
        # 워커 프로세스 생성
        for i in range(self.max_workers):
            process = Process(
                target=inference_worker_initialized,
                args=(i, model_cache_path, self.video_queue, self.result_queue, self.stop_event, 
                      self.VIDEO_PROGRESS, self.VIDEO_RESULTS, self.VIDEO_FRAMES, self.EVENTS, 
                      self.METRICS, annotation_data)
            )
            process.start()
            self.worker_processes.append(process)
        
        print(f"워커 프로세스 풀 생성 완료: {self.max_workers}개 프로세스")
    
    def _cleanup_worker_pool(self):
        """워커 프로세스 풀 정리"""
        if self.stop_event:
            self.stop_event.set()
        
        # 종료 신호 전송
        if self.video_queue:
            for _ in range(self.max_workers):
                self.video_queue.put(None)
        
        # 프로세스 종료 대기
        for process in self.worker_processes:
            if process.is_alive():
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
        
        self.worker_processes.clear()
        print("워커 프로세스 풀 정리 완료")
    
    def load_model(self, model_id: str, annotation_data: Optional[Dict] = None):
        try:
            print(f"모델 캐시 준비 시작: {model_id}")
            # 모델을 캐시에 다운로드 (한 번만)
            self.current_model_cache_path = snapshot_download(model_id, local_files_only=False)
            self.current_model_id = model_id
            print(f"모델 캐시 준비 완료: {model_id}, 경로: {self.current_model_cache_path}")
            
            # 워커 프로세스 풀 생성 (모델 로딩 포함)
            self._create_worker_pool(self.current_model_cache_path, annotation_data)
            
        except Exception as e:
            print(f"모델 캐시 준비 실패: {e}")
            raise

    def start_inference(self, video_files: List[Path], interval: int, infer_period: int, 
                       batch: int, model_id: str, annotation_data: Optional[Dict] = None):
        def run_inference():
            # 기존 start_inference의 Pool/결과 수집 로직을 이 함수로 이동
            if self.current_model_cache_path is None or self.current_model_id != model_id:
                self.load_model(model_id, annotation_data)
            
            self.IS_INFERENCING.value = True
            self.TOTAL_VIDEOS.value = len(video_files)
            self.PROCESSED_VIDEOS.value = 0
            self.STOP_FLAG.value = False
            self.VIDEO_STATES.clear()
            self.VIDEO_PROGRESS.clear()
            self.VIDEO_FRAMES.clear()
            self.VIDEO_RESULTS.clear()
            self.EVENTS[:] = []
            self.CUMULATIVE_ACCURACY.value = 0.0
            for key in self.METRICS:
                self.METRICS[key] = 0.0
            
            # 비디오 상태 초기화
            for video_path in video_files:
                self.VIDEO_STATES[video_path.name] = "queued"
                self.VIDEO_PROGRESS[video_path.name] = 0.0
            
            # 비디오 작업을 큐에 추가
            if self.video_queue is not None:
                for video_path in video_files:
                    video_task = (str(video_path), interval, infer_period, batch, annotation_data, str(self.results_dir))
                    self.video_queue.put(video_task)
                    self.VIDEO_STATES[video_path.name] = "processing"
            
            # 결과 수집 (실시간 업데이트)
            results = []
            completed_count = 0
            
            while completed_count < len(video_files) and self.result_queue is not None:
                try:
                    result = self.result_queue.get(timeout=1)
                    results.append(result)
                    completed_count += 1
                    self.PROCESSED_VIDEOS.value = completed_count
                    
                    # 비디오 상태 업데이트
                    video_name = result.get("video_name", "unknown")
                    if result.get("status") == "completed":
                        self.VIDEO_STATES[video_name] = "completed"
                        self.VIDEO_PROGRESS[video_name] = 100.0
                        print(f"비디오 처리 완료: {video_name} (결과: {result.get('results_count', 0)}개)")
                        
                        # 실시간 이벤트 기록
                        self.EVENTS.append({
                            "type": "video_completed",
                            "video_name": video_name,
                            "results_count": result.get('results_count', 0),
                            "timestamp": datetime.now().isoformat()
                        })
                        
                    elif result.get("status") == "error":
                        self.VIDEO_STATES[video_name] = "error"
                        self.VIDEO_PROGRESS[video_name] = 0.0
                        print(f"비디오 처리 오류: {video_name} - {result.get('error', 'Unknown error')}")
                        
                        # 실시간 이벤트 기록
                        self.EVENTS.append({
                            "type": "video_error",
                            "video_name": video_name,
                            "error": result.get('error', 'Unknown error'),
                            "timestamp": datetime.now().isoformat()
                        })
                        
                except queue.Empty:
                    # 타임아웃 - 계속 대기
                    continue
            
            self.IS_INFERENCING.value = False
            print(f"모든 비디오 처리 완료: {len(results)}개")
            return results
        # 메인 프로세스 블로킹 방지: 백그라운드 스레드에서 실행
        thread = threading.Thread(target=run_inference, daemon=True)
        thread.start()

    def stop_inference(self):
        """추론 중지 및 결과 저장"""
        self.STOP_FLAG.value = True
        self._cleanup_worker_pool()
        
        # 결과 파일 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = self.save_results_dir / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for f in self.results_dir.glob("*"):
            shutil.move(str(f), str(save_dir / f.name))

    def get_state(self) -> Dict:
        def to_builtin(obj):
            if isinstance(obj, dict):
                return {k: to_builtin(v) for k, v in obj.items()}
            elif hasattr(obj, 'items'):
                return {k: to_builtin(v) for k, v in dict(obj).items()}
            elif isinstance(obj, list):
                return [to_builtin(v) for v in obj]
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, dict)):
                return [to_builtin(v) for v in list(obj)]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        # 실시간 차트/그래프용 데이터 준비
        completed_videos = [name for name, state in self.VIDEO_STATES.items() if state == "completed"]
        processing_videos = [name for name, state in self.VIDEO_STATES.items() if state == "processing"]
        error_videos = [name for name, state in self.VIDEO_STATES.items() if state == "error"]
        
        # 실시간 진행률 데이터
        progress_data = {
            name: {
                "progress": self.VIDEO_PROGRESS.get(name, 0.0),
                "state": self.VIDEO_STATES.get(name, "unknown")
            }
            for name in self.VIDEO_STATES.keys()
        }
        
        # 실시간 결과 데이터
        results_summary = {}
        for name in completed_videos:
            video_results = self.VIDEO_RESULTS.get(name, [])
            if video_results:
                results_summary[name] = {
                    "total_results": len(video_results),
                    "fight_count": sum(1 for r in video_results if isinstance(r, dict) and r.get('prediction_label') == 'Fight'),
                    "nonfight_count": sum(1 for r in video_results if isinstance(r, dict) and r.get('prediction_label') == 'NonFight'),
                    "avg_inference_time": sum(r.get('inference_time_ms', 0) for r in video_results if isinstance(r, dict)) / len(video_results) if video_results else 0,
                    "avg_inference_fps": sum(r.get('inference_fps', 0) for r in video_results if isinstance(r, dict)) / len(video_results) if video_results else 0
                }
        
        # 현재 추론 중인 비디오 및 진행률
        current_video = None
        current_progress = 0.0
        for name, state in self.VIDEO_STATES.items():
            if state == "processing":
                current_video = name
                current_progress = self.VIDEO_PROGRESS.get(name, 0.0)
                break
        per_video_progress = dict(self.VIDEO_PROGRESS)
        
        return {
            "is_inferencing": self.IS_INFERENCING.value,
            "total_videos": self.TOTAL_VIDEOS.value,
            "processed_videos": self.PROCESSED_VIDEOS.value,
            "video_states": to_builtin(self.VIDEO_STATES),
            "video_progress": to_builtin(self.VIDEO_PROGRESS),
            "video_frames": list(self.VIDEO_FRAMES.keys()),
            "cumulative_accuracy": self.CUMULATIVE_ACCURACY.value,
            "metrics": to_builtin(self.METRICS),
            "events": to_builtin(self.EVENTS),
            # 실시간 차트/그래프용 추가 데이터
            "progress_data": progress_data,
            "results_summary": results_summary,
            "completed_videos": completed_videos,
            "processing_videos": processing_videos,
            "error_videos": error_videos,
            "stream_video": self._get_random_stream_video(),
            "overall_progress": (len(completed_videos) / self.TOTAL_VIDEOS.value * 100) if self.TOTAL_VIDEOS.value > 0 else 0,
            # 프론트엔드 호환 필드
            "current_video": current_video,
            "current_progress": current_progress,
            "per_video_progress": per_video_progress
        }

    def get_video_frame(self, video_name: str) -> Optional[bytes]:
        return self.VIDEO_FRAMES.get(video_name)

    def cleanup(self):
        self.stop_inference()
        self._cleanup_worker_pool()
        if self.manager:
            self.manager.shutdown()
            self.manager = None

    def _get_random_stream_video(self) -> Optional[str]:
        """랜덤하게 스트림용 비디오 선택"""
        if not self.VIDEO_STATES:
            return None
        
        # 처리 중인 비디오 중에서 랜덤 선택
        processing_videos = [name for name, state in self.VIDEO_STATES.items() 
                           if state in ["processing", "queued"]]
        
        if processing_videos:
            import random
            return random.choice(processing_videos)
        
        return None

def _update_metrics_static(video_name: str, results: list, annotation: dict, metrics_dict: Dict):
    tp = tn = fp = fn = 0
    gt_label = annotation.get('label') if isinstance(annotation, dict) else None
    if gt_label is None:
        return
    for result in results:
        pred = result['prediction_label']
        if pred == 'Fight' and gt_label == 'Fight':
            tp += 1
        elif pred == 'NonFight' and gt_label == 'NonFight':
            tn += 1
        elif pred == 'Fight' and gt_label == 'NonFight':
            fp += 1
        elif pred == 'NonFight' and gt_label == 'Fight':
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    metrics_dict['tp'] = metrics_dict.get('tp', 0) + tp
    metrics_dict['tn'] = metrics_dict.get('tn', 0) + tn
    metrics_dict['fp'] = metrics_dict.get('fp', 0) + fp
    metrics_dict['fn'] = metrics_dict.get('fn', 0) + fn
    metrics_dict['precision'] = precision
    metrics_dict['recall'] = recall
    metrics_dict['f1_score'] = f1 
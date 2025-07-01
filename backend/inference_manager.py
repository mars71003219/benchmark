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
import pandas as pd

# 멀티프로세싱 시작 방식 설정 (spawn 방식으로 설정)
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 이미 설정된 경우 무시

from utils import load_model, process_video, create_overlay_video, create_results_csv

def inference_worker_initialized(worker_id: int, model_cache_path: str, video_queue, result_queue, stop_event, shared_events, annotation_data=None, mode="AR"):
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
    
    stream_video_name = None
    
    while not stop_event.is_set():
        try:
            video_task = video_queue.get(timeout=5)
            if video_task is None:
                break
            video_path_str, interval, infer_period, batch, task_annotation_data, results_dir_str = video_task
            video_path = Path(video_path_str)
            results_dir = Path(results_dir_str)
            print(f"워커 {worker_id}: 비디오 처리 시작 - {video_path.name}")
            if stream_video_name is None:
                stream_video_name = video_path.name
            try:
                from utils import process_video, create_overlay_video
                def progress_callback(current, total):
                    progress_percent = (current / total) * 100 if total > 0 else 0
                    shared_events.append({
                        "type": "progress_update",
                        "video_name": video_path.name,
                        "current": current,
                        "total": total,
                        "progress": progress_percent,
                        "timestamp": datetime.now().isoformat()
                    })
                def result_callback(result):
                    shared_events.append({
                        "type": "inference_result",
                        "video_name": video_path.name,
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    })
                def stop_checker():
                    return stop_event.is_set()
                def frame_callback(frame_data):
                    if video_path.name == stream_video_name:
                        shared_events.append({
                            "type": "frame_update",
                            "video_name": video_path.name,
                            "timestamp": datetime.now().isoformat()
                        })
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
                if results:
                    overlay_path = results_dir / f"{video_path.stem}_overlay.mp4"
                    create_overlay_video(video_path, results, overlay_path)
                result = {
                    "worker_id": worker_id,
                    "video_name": video_path.name,
                    "status": "completed",
                    "results_count": len(results) if results else 0,
                    "overlay_path": str(overlay_path) if results else None,
                    "results": results
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
            continue
        except Exception as e:
            print(f"워커 {worker_id}: 오류 발생 - {e}")
            result_queue.put({
                "worker_id": worker_id,
                "error": str(e),
                "status": "error"
            })
    del model
    del feature_extractor
    torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
    print(f"워커 {worker_id}: 종료")

class InferenceManager:
    """멀티프로세싱 기반 추론 관리자 (프로세스 풀 + 큐 방식)"""
    def __init__(self, upload_dir: Path, results_dir: Path, save_results_dir: Path,
                 manager=None,
                 is_inferencing=None, total_videos=None, processed_videos=None, stop_flag=None,
                 cumulative_accuracy=None, events=None):
        self.upload_dir = upload_dir
        self.results_dir = results_dir
        self.save_results_dir = save_results_dir
        self.current_model_id = None
        self.current_model_cache_path = None
        self.worker_processes = []
        self.video_queue = None
        self.result_queue = None
        self.stop_event = None
        self.max_workers = 4
        self.manager = manager or mp.Manager()
        self.IS_INFERENCING = is_inferencing or self.manager.Value('b', False)
        self.TOTAL_VIDEOS = total_videos or self.manager.Value('i', 0)
        self.PROCESSED_VIDEOS = processed_videos or self.manager.Value('i', 0)
        self.STOP_FLAG = stop_flag or self.manager.Value('b', False)
        self.CUMULATIVE_ACCURACY = cumulative_accuracy or self.manager.Value('d', 0.0)
        self.EVENTS = events or self.manager.list()
        self.all_results = []
        self.video_results = {}  # 메인스레드에서만 관리
        # 비디오별 최종 결과 누적 리스트
        self.video_final_results = []
        # 구간별 추론 결과 누적 리스트
        self.segment_results = []
        # 서버 시작 시 final_results.csv, results.csv에서 복원
        final_csv_path = self.results_dir / "final_results.csv"
        if final_csv_path.exists():
            try:
                df = pd.read_csv(final_csv_path)
                self.video_final_results = df.to_dict(orient="records")
            except Exception:
                self.video_final_results = []
        results_csv_path = self.results_dir / "results.csv"
        if results_csv_path.exists():
            try:
                df = pd.read_csv(results_csv_path)
                self.segment_results = df.to_dict(orient="records")
            except Exception:
                self.segment_results = []
    
    def _create_worker_pool(self, model_cache_path: str, annotation_data: Optional[Dict] = None, mode: str = "AR"):
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
                      self.EVENTS, annotation_data, mode)
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
    
    def load_model(self, model_id: str, annotation_data: Optional[Dict] = None, mode: str = "AR"):
        try:
            print(f"모델 캐시 준비 시작: {model_id}")
            # 모델을 캐시에 다운로드 (한 번만)
            self.current_model_cache_path = snapshot_download(model_id, local_files_only=False)
            self.current_model_id = model_id
            print(f"모델 캐시 준비 완료: {model_id}, 경로: {self.current_model_cache_path}")
            
            # 워커 프로세스 풀 생성 (모델 로딩 포함)
            self._create_worker_pool(self.current_model_cache_path, annotation_data, mode)
            
        except Exception as e:
            print(f"모델 캐시 준비 실패: {e}")
            raise

    def update_metrics_from_results(self, all_results, annotation_data, mode="AR", min_consecutive=7):
        tp = tn = fp = fn = 0
        final_rows = []
        video_final_results = []
        for result in all_results:
            video_name = result.get('video_name')
            pred_labels = [r['prediction_label'] for r in result.get('results', [])]
            is_fight = is_consecutive_fight(pred_labels, min_consecutive)
            final_pred_label = "Fight" if is_fight else "NonFight"
            gt_label = None
            if annotation_data:
                if mode in annotation_data.get(video_name, {}):
                    gt_label = annotation_data.get(video_name, {}).get(mode, {}).get('label')
                else:
                    gt_label = annotation_data.get(video_name, {}).get('label')
            if gt_label is None:
                continue
            if final_pred_label == "Fight" and gt_label == "Fight":
                metric = "TP"
                tp += 1
            elif final_pred_label == "Fight" and gt_label == "NonFight":
                metric = "FP"
                fp += 1
            elif final_pred_label == "NonFight" and gt_label == "Fight":
                metric = "FN"
                fn += 1
            elif final_pred_label == "NonFight" and gt_label == "NonFight":
                metric = "TN"
                tn += 1
            else:
                metric = ""
            row = {
                "video_name": video_name,
                "final_pred_label": final_pred_label,
                "anno_label": gt_label,
                "metrics": metric
            }
            final_rows.append(row)
            video_final_results.append(row)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # final_results.csv 저장 (비디오별 1개 row)
        print(f"[DEBUG] final_results.csv 저장 경로: {final_csv_path}")
        print(f"[DEBUG] final_rows 데이터: {final_rows}")
        print(f"[DEBUG] results_dir exists? {self.results_dir.exists()}")
        df = pd.DataFrame(final_rows)
        print(f"[DEBUG] DataFrame empty? {df.empty}")
        df.to_csv(final_csv_path, index=False)
        metrics = {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "accuracy": accuracy, "f1_score": f1
        }
        return metrics, video_final_results

    def update_metrics_from_results_single_video(self, video_name, results, annotation_data, mode="AR", min_consecutive=7):
        pred_labels = [r['prediction_label'] for r in results]
        is_fight = is_consecutive_fight(pred_labels, min_consecutive)
        final_pred_label = "Fight" if is_fight else "NonFight"
        gt_label = None
        if annotation_data:
            if mode in annotation_data.get(video_name, {}):
                gt_label = annotation_data.get(video_name, {}).get(mode, {}).get('label')
            else:
                gt_label = annotation_data.get(video_name, {}).get('label')
        metric = ""
        if gt_label is not None:
            if final_pred_label == "Fight" and gt_label == "Fight":
                metric = "TP"
            elif final_pred_label == "Fight" and gt_label == "NonFight":
                metric = "FP"
            elif final_pred_label == "NonFight" and gt_label == "Fight":
                metric = "FN"
            elif final_pred_label == "NonFight" and gt_label == "NonFight":
                metric = "TN"
        else:
            gt_label = "N/A"
        row = {
            "video_name": video_name,
            "final_pred_label": final_pred_label,
            "anno_label": gt_label,
            "metrics": metric
        }
        # 메모리 누적 리스트에 갱신(동일 비디오 있으면 교체, 없으면 append)
        self.video_final_results = [r for r in self.video_final_results if r["video_name"] != video_name]
        self.video_final_results.append(row)
        # 구간별 추론 결과 누적 (results: 구간별 리스트)
        # 기존 비디오 구간 결과 삭제 후 append
        self.segment_results = [r for r in self.segment_results if r.get("video_name") != video_name]
        for seg in results:
            self.segment_results.append(seg)
        # results.csv에도 갱신
        results_csv_path = self.results_dir / "results.csv"
        try:
            df = pd.DataFrame(self.segment_results)
            df.to_csv(results_csv_path, index=False)
        except Exception:
            pass
        # final_results.csv에도 갱신
        final_csv_path = self.results_dir / "final_results.csv"
        try:
            df = pd.DataFrame(self.video_final_results)
            df.to_csv(final_csv_path, index=False)
        except Exception:
            pass
        # 웹소켓 이벤트로도 즉시 전송
        self.EVENTS.append({
            "type": "video_final_result",
            "video_name": video_name,
            "final_result": row,
            "timestamp": datetime.now().isoformat()
        })

    def start_inference(self, video_files: List[Path], interval: int, infer_period: int, 
                       batch: int, model_id: str, annotation_data: Optional[Dict] = None, mode: str = "AR"):
        def run_inference():
            if self.current_model_cache_path is None or self.current_model_id != model_id:
                self.load_model(model_id, annotation_data, mode)
            self.IS_INFERENCING.value = True
            self.TOTAL_VIDEOS.value = len(video_files)
            self.PROCESSED_VIDEOS.value = 0
            self.STOP_FLAG.value = False
            self.EVENTS[:] = []
            self.all_results = []
            self.video_results = {}
            for video_path in video_files:
                self.video_results[video_path.name] = []
            if self.video_queue is not None:
                for video_path in video_files:
                    video_task = (str(video_path), interval, infer_period, batch, annotation_data, str(self.results_dir))
                    self.video_queue.put(video_task)
            results = []
            completed_count = 0
            max_wait = 60 * len(video_files)  # 비디오당 60초 제한
            wait_count = 0
            while completed_count < len(video_files) and self.result_queue is not None:
                try:
                    result = self.result_queue.get(timeout=1)
                    print(f"[DEBUG] result_queue.get: {result}")
                    results.append(result)
                    completed_count += 1
                    print(f"[DEBUG] completed_count: {completed_count}, total: {len(video_files)}")
                    video_name = result.get("video_name", "unknown")
                    if result.get("status") == "completed" and result.get("results"):
                        self.video_results[video_name] = result["results"]
                        self.all_results.append({"video_name": video_name, "results": result["results"]})
                        from utils import create_results_csv
                        create_results_csv(self.all_results, self.results_dir / "results.csv")
                        self.EVENTS.append({
                            "type": "video_completed",
                            "video_name": video_name,
                            "results_count": result.get('results_count', 0),
                            "timestamp": datetime.now().isoformat()
                        })
                        self.video_results[video_name] = []
                        # 비디오별로 final_results.csv 업데이트
                        self.update_metrics_from_results_single_video(video_name, result["results"], annotation_data, mode)
                    elif result.get("status") == "error":
                        self.EVENTS.append({
                            "type": "video_error",
                            "video_name": video_name,
                            "error": result.get('error', 'Unknown error'),
                            "timestamp": datetime.now().isoformat()
                        })
                except queue.Empty:
                    wait_count += 1
                    if wait_count > max_wait:
                        print("[ERROR] result_queue에서 결과를 모두 받지 못하고 타임아웃!")
                        break
                    continue
            # 모든 결과가 모이면 전체 혼동행렬/정확도 등은 update_metrics_from_results로 계산(필요시)
            metrics, video_final_results = self.update_metrics_from_results(self.all_results, annotation_data, mode)
            # 누적 정확도 업데이트
            prev_acc = self.CUMULATIVE_ACCURACY.value
            if prev_acc == 0.0:
                self.CUMULATIVE_ACCURACY.value = metrics["accuracy"]
            else:
                self.CUMULATIVE_ACCURACY.value = (prev_acc + metrics["accuracy"]) / 2
            # 웹소켓 이벤트로 최종 메트릭, 비디오별 결과, 누적 정확도 전송
            self.EVENTS.append({
                "type": "final_metrics",
                "metrics": metrics,
                "video_final_results": video_final_results,
                "cumulative_accuracy": self.CUMULATIVE_ACCURACY.value,
                "timestamp": datetime.now().isoformat()
            })
            self.IS_INFERENCING.value = False
            print(f"모든 비디오 처리 완료: {len(results)}개")
            return results
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
        # video_final_results, segment_results를 항상 반환
        video_final_results = self.video_final_results
        segment_results = self.segment_results
        completed_videos = [name for name, results in self.video_results.items() if results]
        results_summary = {}
        for name in completed_videos:
            video_results = self.video_results.get(name, [])
            if video_results:
                results_summary[name] = {
                    "total_results": len(video_results),
                    "fight_count": sum(1 for r in video_results if isinstance(r, dict) and r.get('prediction_label') == 'Fight'),
                    "nonfight_count": sum(1 for r in video_results if isinstance(r, dict) and r.get('prediction_label') == 'NonFight'),
                    "avg_inference_time": sum(r.get('inference_time_ms', 0) for r in video_results if isinstance(r, dict)) / len(video_results) if video_results else 0,
                    "avg_inference_fps": sum(r.get('inference_fps', 0) for r in video_results if isinstance(r, dict)) / len(video_results) if video_results else 0
                }
        return {
            "is_inferencing": self.IS_INFERENCING.value,
            "total_videos": self.TOTAL_VIDEOS.value,
            "processed_videos": self.PROCESSED_VIDEOS.value,
            "cumulative_accuracy": self.CUMULATIVE_ACCURACY.value,
            "video_final_results": video_final_results,
            "segment_results": segment_results,
            "events": to_builtin(self.EVENTS),
            "results_summary": results_summary,
            "video_results": to_builtin(self.video_results),
        }

    def cleanup(self):
        self.stop_inference()
        self._cleanup_worker_pool()
        if self.manager:
            self.manager.shutdown()
            self.manager = None

# 연속 Fight 판정 함수 (최소 연속값 디폴트 7)
def is_consecutive_fight(pred_list, min_consecutive=7):
    count = 0
    for label in pred_list:
        if label == "Fight":
            count += 1
            if count >= min_consecutive:
                return True
        else:
            count = 0
    return False 
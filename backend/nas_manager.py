import os
import shutil
from pathlib import Path
from typing import Dict, List
import threading

class NASManager:
    """NAS 폴더 관리자"""
    
    def __init__(self, upload_dir: Path, nas_base_path: str, nas_target_path: str):
        self.upload_dir = upload_dir
        self.nas_base_path = nas_base_path
        self.nas_target_path = nas_target_path
        self.copy_progress = {
            "is_copying": False,
            "total_files": 0,
            "copied_files": 0,
            "current_file": None,
            "errors": []
        }
        self.copy_lock = threading.Lock()
    
    def reset_progress(self):
        """복사 진행 상태 초기화"""
        with self.copy_lock:
            self.copy_progress = {
                "is_copying": False,
                "total_files": 0,
                "copied_files": 0,
                "current_file": None,
                "errors": []
            }
    
    def copy_nas_folder(self, nas_folder_path: str):
        """NAS 폴더의 모든 비디오 파일을 uploads로 복사"""
        with self.copy_lock:
            self.reset_progress()
            self.copy_progress["is_copying"] = True
        
        try:
            # 경로 보안 검증 및 변환
            nas_path = self._validate_and_transform_path(nas_folder_path)
            
            # 비디오 파일 찾기
            video_files = self._find_video_files(nas_path)
            
            with self.copy_lock:
                self.copy_progress["total_files"] = len(video_files)
            
            if self.copy_progress["total_files"] == 0:
                with self.copy_lock:
                    self.copy_progress["errors"].append("선택된 폴더에서 비디오 파일을 찾을 수 없습니다.")
                return
            
            # 파일 복사
            skipped_files = []
            for video_file in video_files:
                if not self.copy_progress["is_copying"]:
                    break
                
                try:
                    with self.copy_lock:
                        self.copy_progress["current_file"] = str(video_file)
                    
                    target_filename = video_file.name
                    target_path = self.upload_dir / target_filename
                    
                    # 중복 파일명 처리
                    counter = 1
                    original_target_path = target_path
                    while target_path.exists():
                        name_without_ext = original_target_path.stem
                        ext = original_target_path.suffix
                        target_filename = f"{name_without_ext}_{counter}{ext}"
                        target_path = self.upload_dir / target_filename
                        counter += 1
                    
                    if target_path.exists():
                        skipped_files.append(str(target_path.name))
                        with self.copy_lock:
                            self.copy_progress["copied_files"] += 1
                        continue
                    
                    shutil.copy2(video_file, target_path)
                    
                    with self.copy_lock:
                        self.copy_progress["copied_files"] += 1
                        
                except Exception as e:
                    error_msg = f"파일 복사 실패 ({video_file.name}): {str(e)}"
                    with self.copy_lock:
                        self.copy_progress["errors"].append(error_msg)
            
            if skipped_files:
                with self.copy_lock:
                    self.copy_progress["errors"].append(f"이미 존재하여 건너뛴 파일: {', '.join(skipped_files)}")
                    
        except Exception as e:
            error_msg = f"NAS 폴더 처리 중 오류 발생: {str(e)}"
            with self.copy_lock:
                self.copy_progress["errors"].append(error_msg)
        finally:
            with self.copy_lock:
                self.copy_progress["is_copying"] = False
    
    def _validate_and_transform_path(self, nas_folder_path: str) -> Path:
        """경로 검증 및 변환"""
        try:
            if nas_folder_path.startswith(self.nas_base_path):
                nas_folder_path = nas_folder_path.replace(self.nas_base_path, self.nas_target_path, 1)
            elif nas_folder_path.startswith('/home'):
                nas_folder_path = nas_folder_path.replace('/home', self.nas_target_path, 1)
            
            if not nas_folder_path.startswith('/'):
                nas_path = Path.cwd() / nas_folder_path
            else:
                nas_path = Path(nas_folder_path)
            
            nas_path = nas_path.resolve()
            
            if not str(nas_path).startswith(self.nas_target_path):
                raise ValueError(f"허용되지 않은 경로입니다. {self.nas_target_path}로 시작하는 경로만 사용 가능합니다.")
            
            if not nas_path.exists():
                raise ValueError(f"경로가 존재하지 않습니다: {nas_folder_path}")
            
            if not nas_path.is_dir():
                raise ValueError(f"지정된 경로가 디렉토리가 아닙니다: {nas_folder_path}")
            
            return nas_path
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise ValueError(f"경로 검증 실패: {str(e)}")
    
    def _find_video_files(self, nas_path: Path) -> List[Path]:
        """비디오 파일 찾기"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(nas_path.rglob(f"*{ext}"))
            video_files.extend(nas_path.rglob(f"*{ext.upper()}"))
        
        video_files = list(set(video_files))
        video_files.sort()
        return video_files
    
    def get_progress(self) -> Dict:
        """복사 진행 상태 반환"""
        with self.copy_lock:
            return self.copy_progress.copy()
    
    def cancel_copy(self):
        """복사 취소"""
        with self.copy_lock:
            self.copy_progress["is_copying"] = False
    
    def get_nas_paths(self) -> Dict[str, str]:
        """NAS 경로 설정 반환"""
        return {
            "base_path": self.nas_base_path,
            "target_path": self.nas_target_path
        } 
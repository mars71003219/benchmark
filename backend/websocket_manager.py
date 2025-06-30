# /backend/websocket_manager.py

import asyncio
import json
import base64
from typing import Dict, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

class WebSocketManager:
    """웹소켓 연결 관리자"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.overlay_connections: Dict[str, WebSocket] = {}  # video_name -> websocket
        self.current_stream_video: Optional[str] = None
    
    async def connect(self, websocket: WebSocket):
        """웹소켓 연결 추가"""
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """웹소켓 연결 제거"""
        self.active_connections.discard(websocket)
        # 오버레이 연결에서도 제거
        for video_name, ws in list(self.overlay_connections.items()):
            if ws == websocket:
                del self.overlay_connections[video_name]
    
    async def broadcast_state(self, state: Dict):
        """모든 연결에 상태 브로드캐스트"""
        if not self.active_connections:
            return
        
        message = json.dumps(state)
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                disconnected.add(connection)
            except Exception as e:
                print(f"웹소켓 전송 오류: {e}")
                disconnected.add(connection)
        
        # 연결 해제된 웹소켓 제거
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_frame(self, video_name: str, frame_data: bytes):
        """특정 비디오의 프레임을 해당 오버레이 연결에 전송"""
        if video_name in self.overlay_connections:
            try:
                frame_base64 = base64.b64encode(frame_data).decode('utf-8')
                await self.overlay_connections[video_name].send_text(frame_base64)
            except WebSocketDisconnect:
                del self.overlay_connections[video_name]
            except Exception as e:
                print(f"프레임 전송 오류: {e}")
                del self.overlay_connections[video_name]
    
    def set_overlay_connection(self, video_name: str, websocket: WebSocket):
        """오버레이 스트림 연결 설정"""
        self.overlay_connections[video_name] = websocket
        self.current_stream_video = video_name
    
    def get_current_stream_video(self) -> Optional[str]:
        """현재 스트리밍 중인 비디오 이름 반환"""
        return self.current_stream_video
    
    def get_available_videos(self) -> Set[str]:
        """오버레이 연결이 가능한 비디오 목록 반환"""
        return set(self.overlay_connections.keys())

# 글로벌 웹소켓 매니저 인스턴스
websocket_manager = WebSocketManager() 
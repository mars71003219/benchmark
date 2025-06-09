// frontend/src/hooks/useWebSocket.ts

import { useState, useEffect, useRef } from 'react';

// 백엔드에서 오는 데이터 구조와 일치하는 타입 정의
interface InferenceEvent {
  type: string;
  video?: string;
  message?: string;
  timestamp: string;
  results?: number;
  total_results?: number;
}

export interface InferenceState {
  total_videos: number;
  processed_videos: number;
  current_video: string | null;
  current_progress: number;
  events: InferenceEvent[];
  per_video_progress: { [key: string]: number };
  is_inferencing: boolean;
}

// 훅의 기본 상태값
const initialState: InferenceState = {
  total_videos: 0,
  processed_videos: 0,
  current_video: null,
  current_progress: 0,
  events: [],
  per_video_progress: {},
  is_inferencing: false,
};

// 이제 훅은 연결 중일 때 null을 반환할 수 있음
export const useWebSocket = (url: string): InferenceState | null => {
  const [data, setData] = useState<InferenceState | null>(null); // 초기 상태를 null로 변경
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    const connect = () => {
      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        console.log("WebSocket connected");
        // 연결 성공 시 초기 상태로 설정
        setData(initialState); 
      };

      ws.current.onmessage = (event) => {
        try {
          const parsedData = JSON.parse(event.data);
          setData(parsedData);
        } catch (error) {
          console.error("Failed to parse WebSocket message:", error);
        }
      };

      ws.current.onerror = (error) => {
        console.error("WebSocket error:", error);
      };

      ws.current.onclose = () => {
        console.log("WebSocket disconnected, attempting to reconnect...");
        setData(null); // 연결 종료 시 다시 null로 설정
        setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      if (ws.current) {
        ws.current.onclose = null; // 재연결 로직 방지
        ws.current.close();
      }
    };
  }, [url]);

  return data;
};
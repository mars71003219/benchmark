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
  cumulative_accuracy: number;
  metrics: {
    tp: number;
    tn: number;
    fp: number;
    fn: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
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
  cumulative_accuracy: 0.0,
  metrics: { tp: 0, tn: 0, fp: 0, fn: 0, precision: 0.0, recall: 0.0, f1_score: 0.0 },
};

// 이제 훅은 연결 중일 때 null을 반환할 수 있음
export const useWebSocket = (url: string): InferenceState | null => {
  const [data, setData] = useState<InferenceState | null>(null);
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout>();
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const baseDelay = 1000; // 1초

  const connect = () => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        console.log("WebSocket connected");
        reconnectAttempts.current = 0;
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

      ws.current.onclose = (event) => {
        console.log(`WebSocket disconnected (code: ${event.code}, reason: ${event.reason})`);
        setData(null);

        // 정상적인 종료가 아닌 경우에만 재연결 시도
        if (event.code !== 1000 && event.code !== 1001) {
          if (reconnectAttempts.current < maxReconnectAttempts) {
            const delay = baseDelay * Math.pow(2, reconnectAttempts.current);
            console.log(`Attempting to reconnect in ${delay}ms... (Attempt ${reconnectAttempts.current + 1}/${maxReconnectAttempts})`);
            
            reconnectTimeout.current = setTimeout(() => {
              reconnectAttempts.current += 1;
              connect();
            }, delay);
          } else {
            console.error("Max reconnection attempts reached");
          }
        }
      };
    } catch (error) {
      console.error("Failed to create WebSocket connection:", error);
    }
  };

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
      if (ws.current) {
        ws.current.onclose = null;
        ws.current.close();
      }
    };
  }, [url]);

  return data;
};
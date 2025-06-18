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

interface WebSocketState {
    is_inferencing: boolean;
    current_video: string | null;
    current_progress: number;
    total_videos: number;
    processed_videos: number;
    events: any[];
    cumulative_accuracy: number;
    metrics: any;
}

export function useWebSocket(url: string) {
    const [state, setState] = useState<WebSocketState | null>(null);
    const [ws, setWs] = useState<WebSocket | null>(null);

    useEffect(() => {
        // 절대경로로 WebSocket 연결
        const socket = new WebSocket('ws://192.168.190.4:10000/ws');
        setWs(socket);

        socket.onopen = () => {
            console.log('WebSocket 연결됨');
        };

        socket.onmessage = (event) => {
            if (event.data === "ping") {
                socket.send("pong");
                return;
            }
            try {
                const data = JSON.parse(event.data);
                setState(data);
            } catch (e) {
                console.error('WebSocket 메시지 파싱 오류:', e);
            }
        };

        socket.onerror = (error) => {
            console.error('WebSocket 오류:', error);
        };

        socket.onclose = () => {
            console.log('WebSocket 연결 종료');
            // 3초 후 재연결 시도는 필요시 별도 구현
        };

        return () => {
            // 항상 close() 호출하여 리소스 누수 방지
            try {
                socket.close();
            } catch (e) {
                // 이미 닫힌 경우 무시
            }
        };
    }, [url]);

    return state;
}
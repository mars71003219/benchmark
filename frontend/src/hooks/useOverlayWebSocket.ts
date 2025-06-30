import { useState, useEffect, useRef } from "react";

export function useOverlayWebSocket(url: string) {
  const [frame, setFrame] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let socket: WebSocket;
    let reconnectTimeout: NodeJS.Timeout;

    function connect() {
      // 브라우저에서 프론트엔드 개발 서버(3000)로 연결, 프록시를 통해 backend로 전달
      const backendPort = 10000;
      const wsUrl = url.startsWith("ws")
        ? url
        : `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.hostname}:${backendPort}${url}`;
      socket = new WebSocket(wsUrl);
      wsRef.current = socket;

      socket.onopen = () => {
        console.log("Overlay WebSocket 연결됨");
      };

      socket.onmessage = (event) => {
        if (event.data === "ping") {
          socket.send("pong");
          return;
        }
        setFrame(event.data); // base64 string
      };

      socket.onerror = (error) => {
        console.error("Overlay WebSocket 오류:", error);
      };

      socket.onclose = () => {
        console.log("Overlay WebSocket 연결 종료, 3초 후 재연결 시도");
        reconnectTimeout = setTimeout(connect, 3000);
      };
    }

    connect();

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
    };
  }, [url]);

  return frame;
}

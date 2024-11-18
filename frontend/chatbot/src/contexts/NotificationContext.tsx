// src/contexts/NotificationContext.tsx
import { BACKEND_URL } from '@configs/config';
import { eventType } from '@util/notification';
import React, { createContext, useContext, useEffect, useState } from 'react';
import { toast } from 'react-toastify'; // 알림 표시를 위한 라이브러리

interface NotificationEvent {
  event: string;
  message: string;
  timestamp: string;
  data: Record<string, object>;
}

interface NotificationContextType {
  connect: (userId: string) => void;
  disconnect: () => void;
  connected: boolean;
}

const NotificationContext = createContext<NotificationContextType>({
  connect: () => {},
  disconnect: () => {},
  connected: false,
});

export const useNotification = () => useContext(NotificationContext);

export const NotificationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [eventSource, setEventSource] = useState<EventSource | null>(null);
  const [connected, setConnected] = useState(false);

  const connect = (userId: string) => {
    console.log("연결 시도...");
    
    if (eventSource) {
      eventSource.close();
    }

    const newEventSource = new EventSource(`${BACKEND_URL}/api/notifications/subscribe/${userId}`, 
      {
        withCredentials: true  // HTTPS에서 쿠키/인증 정보 전송 허용
      }
    );

    console.log("event source : " + newEventSource);
    

    // 연결 이벤트 리스너
    newEventSource.addEventListener('connect', (event) => {
      console.log('Connected to notification stream:', event.data);
      setConnected(true);
    });

    // Vector DB 업데이트 이벤트 리스너
    newEventSource.addEventListener(eventType.UPDATE_VECTOR_DB, (event) => {
      const data: NotificationEvent = JSON.parse(event.data);
      console.log("update : ", data);
      handleVectorDBUpdate(data);
    });

    // 파일 요약 이벤트 리스너
    newEventSource.addEventListener(eventType.SUMMARY_FILE, (event) => {
      const data: NotificationEvent = JSON.parse(event.data);
      console.log("summary : ", data);
      handleFileSummary(data);
    });

    // 에러 처리
    newEventSource.onerror = (error) => {
      console.error('SSE error:', error);
      setConnected(false);
      newEventSource.close();
    };

    setEventSource(newEventSource);
  };

  const disconnect = () => {
    if (eventSource) {
      eventSource.close();
      setEventSource(null);
      setConnected(false);
    }
  };

  // 컴포넌트 언마운트 시 연결 종료
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, []);

  // 이벤트 처리 함수들
  const handleVectorDBUpdate = (data: NotificationEvent) => {
    const { message, data: eventData } = data;
    
    toast.info(
      <div>
        <div>{message}</div>
        <div className="truncate overflow-ellipsis text-sm text-gray-600">{`${eventData["title"]}`}</div>
      </div>
    );
  };

  const handleFileSummary = (data: NotificationEvent) => {
    const { message, data: eventData } = data;
    
    toast.info(
      <div>
        <div>{message}</div>
        <div className="truncate  overflow-ellipsis text-sm text-gray-600">{`${eventData["fileName"]}`}</div>
      </div>
      );
  };

  return (
    <NotificationContext.Provider value={{ connect, disconnect, connected }}>
      {children}
    </NotificationContext.Provider>
  );
};
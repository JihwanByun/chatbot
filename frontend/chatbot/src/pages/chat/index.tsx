import React, { useCallback, useEffect, useState } from "react";
import ChatView, { Doc, QnA } from "@components/chat/ChatView";
import ButtonWithIcon from "@components/atoms/button/ButtonWithIcon";
import { BaseMessage } from "@components/chat/ChatMessage";
import SembotLayout from "@pages/SembotLayout";
import {
  searchDocsAPI,
  generateAPI,
  getChatroomListAPI,
  ChatroomList,
  ChatroomResponse,
  createChatroomAPI,
  getChatroomDetailAPI,
  ChatroomDetail,
  createChatAPI,
  createChatFeedbackAPI
} from "@apis/chat/chatApi";

export type ButtonWithIconProps = React.ComponentProps<typeof ButtonWithIcon>;

interface ChatState {
  curChatroomId: number;
  currentChatroomPage: number;
  hasNextChatroom: boolean;
  qnas: QnA[];
  isLoading: boolean;
  isFetchingChatrooms: boolean;
  error: string | null;
}

const initialState: ChatState = {
  curChatroomId: -1,
  currentChatroomPage: 0,
  hasNextChatroom: true,
  qnas: [],
  isLoading: false,
  isFetchingChatrooms: false,
  error: null,
};

const Chat: React.FC = () => {
  // State Management
  const [state, setState] = useState<ChatState>(initialState);
  const [chatroomComponents, setChatroomComponents] = useState<ButtonWithIconProps[]>([]);

  // UI Constants
  const footStyle = "flex bg-transparent text-white py-2 px-4 rounded mx-1";
  const newChatButtonStyle = "flex bg-transparent border border-white text-white py-2 px-4 rounded mx-1 hover:bg-blue-900 transition-colors duration-200 ease-in-out";
  const chatroomButtonStyle = "flex bg-white text-semesBlue py-2 px-4 rounded mx-1";

  // Utility Functions
  const createBaseButton = (btnName: string, style: string, icon: string, handleClick?: () => void): ButtonWithIconProps => ({
    btnName,
    styleName: style,
    icon,
    handleClick,
  });

  // Chat Room Management
  const newChatProp = createBaseButton(
    "새채팅",
    newChatButtonStyle,
    "/src/assets/icons/plus.svg",
    () => {
      setState(prev => ({ ...prev, curChatroomId: -1, qnas: [] }));
    }
  );

  const footerComponents: ButtonWithIconProps[] = [
    createBaseButton("규정 확인하기", footStyle, "/src/assets/icons/book-open-text.svg"),
    createBaseButton("로그아웃", footStyle, "/src/assets/icons/logout.svg"),
  ];

  // API Handlers
  const handleApiError = useCallback((error: unknown, customMessage: string) => {
    console.error("Error:", error);
    setState(prev => ({
      ...prev,
      error: error instanceof Error ? error.message : customMessage
    }));
  }, []);

  const fetchChatroom = useCallback(async (chatroomId: number) => {
    try {
      const chatroom: ChatroomDetail | null = await getChatroomDetailAPI(chatroomId);
      if (!chatroom) throw new Error('채팅방 정보를 가져오는데 실패했습니다.');

      setState(prev => ({
        ...prev,
        curChatroomId: chatroomId,
        qnas: chatroom.qnas,
        error: null
      }));
    } catch (error) {
      handleApiError(error, '채팅방 정보를 가져오는데 실패했습니다.');
    }
  }, [handleApiError]);

  const fetchChatrooms = useCallback(async () => {
    if (state.isFetchingChatrooms || !state.hasNextChatroom || state.error) return;

    try {
      setState(prev => ({ ...prev, isFetchingChatrooms: true, error: null }));

      const chatroomList: ChatroomList | null = await getChatroomListAPI(state.currentChatroomPage);
      if (!chatroomList) throw new Error('채팅방 목록을 가져오는데 실패했습니다.');

      const newComponents = chatroomList.contents.map((content: ChatroomResponse) =>
        createBaseButton(
          content.title,
          chatroomButtonStyle,
          "src/assets/icons/delete.svg",
          () => fetchChatroom(content.chatRoomId)
        )
      );

      setState(prev => ({
        ...prev,
        hasNextChatroom: chatroomList.hasNext,
        currentChatroomPage: prev.currentChatroomPage + 1,
      }));

      setChatroomComponents(prev => [...prev, ...newComponents]);
    } catch (error) {
      handleApiError(error, '채팅방 목록을 가져오는데 실패했습니다.');
    } finally {
      setState(prev => ({ ...prev, isFetchingChatrooms: false }));
    }
  }, [state.isFetchingChatrooms, state.hasNextChatroom, state.error, state.currentChatroomPage, handleApiError, fetchChatroom]);

  // Message Handling
  const createQnA = (message: string, docs: Doc[]): QnA => ({
    question: new BaseMessage("question", message),
    answer: new BaseMessage("answer", ""),
    docs,
    isAnswered: false,
  });

  const handleStreamResponse = useCallback(async (reader : ReadableStreamDefaultReader<Uint8Array>, decoder: TextDecoder): Promise<string> => {
    let fullAnswer = '';
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = value instanceof Uint8Array ? decoder.decode(value, { stream: true }) : value as string;
      if (chunk) {
        fullAnswer += chunk;
        setState(prev => ({
          ...prev,
          qnas: prev.qnas.map((qna, idx) => 
            idx === prev.qnas.length - 1 
              ? { ...qna, answer: new BaseMessage("answer", fullAnswer)}
              : qna
          )
        }));
      }
    }

    return fullAnswer;
  }, []);

  const handleGenerateResponse = useCallback(async (message: string, activeChatroomId: number, currentQnA: QnA) => {
    try {
      const res = await generateAPI(state.qnas, message);
      if (!res.body) throw new Error("응답 생성에 실패했습니다.");

      const fullAnswer = await handleStreamResponse(
        res.body.getReader(),
        new TextDecoder("utf-8")
      );

      const updatedQnA = {
        ...currentQnA,
        answer: new BaseMessage("answer", fullAnswer)
      };

      const chat = await createChatAPI(activeChatroomId, updatedQnA);
      if (!chat) throw new Error("채팅 저장에 실패했습니다.");

      setState(prev => ({
        ...prev,
        qnas: prev.qnas.map((qna, idx) => 
          idx === prev.qnas.length - 1 
            ? { ...qna, chatId: chat.chatId, isAnswered: true }
            : qna
        )
      }));
    } catch (error) {
      handleApiError(error, "응답 생성에 실패했습니다.");
    }
  }, [state.qnas, handleStreamResponse, handleApiError]);

  const sendMessage = useCallback(async (message: string) => {
    if (state.isLoading || !message) return;

    try {
      setState(prev => ({ ...prev, isLoading: true, error: null }));
      
      let activeChatroomId = state.curChatroomId;
      if (activeChatroomId === -1) {
        const result = await createChatroomAPI(message);
        if (!result) throw new Error("채팅방 생성에 실패했습니다.");
        
        activeChatroomId = result.chatRoomId;
        setState(prev => ({ ...prev, curChatroomId: activeChatroomId }));
        
        const chatroomButton = createBaseButton(
          result.title,
          chatroomButtonStyle,
          "src/assets/icons/delete.svg",
          () => fetchChatroom(result.chatRoomId)
        );
        
        setChatroomComponents(prev => [newChatProp, chatroomButton, ...prev.slice(1)]);
      }

      const docs = await searchDocsAPI(message);
      const newQnA = createQnA(message, docs);
      
      setState(prev => ({
        ...prev,
        qnas: [...prev.qnas, newQnA]
      }));

      await handleGenerateResponse(message, activeChatroomId, newQnA);
    } catch (error) {
      handleApiError(error, "메시지 전송에 실패했습니다.");
    } finally {
      setState(prev => ({ ...prev, isLoading: false }));
    }
  }, [state.isLoading, state.curChatroomId, handleGenerateResponse, handleApiError, fetchChatroom]);

  const handleFeedback = useCallback((qna: QnA, isPositive: boolean, negativeReason? : string) => {
    if(!qna.chatId) return;

    createChatFeedbackAPI(qna.chatId, isPositive, negativeReason);

    setState(prev => ({
      ...prev,
      qnas: prev.qnas.map(q => 
        q.chatId === qna.chatId ? { ...q, isPositive } : q
      )
    }));
  }, []);

  // Initial Load
  useEffect(() => {
    if (state.currentChatroomPage === 0) {
      fetchChatrooms();
    }
  }, [state.currentChatroomPage, fetchChatrooms]);

  useEffect(() => {
    setChatroomComponents([newChatProp]);
  }, []);

  return (
    <SembotLayout
      hideHeader={true}
      isRule={false}
      sidebarComponents={chatroomComponents}
      footerComponents={footerComponents}
      hasMore={state.hasNextChatroom}
      onLoadMore={fetchChatrooms}
      isLoading={state.isFetchingChatrooms}
    >
      <div className="w-full h-full text-left">
        <ChatView
          qnas={state.qnas}
          onSendMessage={sendMessage}
          isLoading={state.isLoading}
          onFeedback={handleFeedback}
        />
      </div>
    </SembotLayout>
  );
};

export default Chat;
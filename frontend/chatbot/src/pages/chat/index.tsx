import React, { useEffect, useState } from "react";
import ChatView, { QnA } from "@components/chat/ChatView";
import ButtonWithIcon from "@components/atoms/button/ButtonWithIcon";
import { BaseMessage } from "@components/chat/ChatMessage";
import SembotLayout from "@pages/SembotLayout";
import { searchDocsAPI, generateAPI, getChatroomListAPI, ChatroomList, ChatroomResponse } from "@apis/chat/chatApi";

export type ButtonWithIconProps = React.ComponentProps<typeof ButtonWithIcon>;

const Chat: React.FC = () => {
  const [curChatroomId, setCurChatroomId] = useState<number>(-1);
  const [currentChatroomPage, setCurrentChatroomPage] = useState<number>(0);
  const [hasNextChatroom, setHasNextChatroom] = useState<boolean>();
  const [qnas, setQnAs] = useState<QnA[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const initChatroom = () => {
    setCurChatroomId(-1);
    setQnAs([]);
  };

  const newChatProp: ButtonWithIconProps = {
    btnName: "새채팅",
    styleName: "flex bg-transparent border border-white text-white py-2 px-4 rounded mx-1 hover:bg-blue-900 transition-colors duration-200 ease-in-out",
    icon: "/src/assets/icons/plus.svg",
    handleClick: initChatroom,
  };

  const [chatroomComponents, setChatroomComponents] = useState<ButtonWithIconProps[]>([newChatProp]);
  
  const fetchChatroom = async (chatroomId : number) => {
    
  }

  const fetchChatrooms = async () => {
    setError(null);

    const chatroomList : ChatroomList | null = await getChatroomListAPI(currentChatroomPage);
    
    if (!chatroomList) {
      throw new Error('채팅방 목록을 가져오는데 실패했습니다.');
    }

    setHasNextChatroom(chatroomList.hasNext);
    setCurrentChatroomPage(currentChatroomPage + 1);

    chatroomList.contents.forEach((content : ChatroomResponse)=>{
    
      const chatroomBtnProp = {
        btnName: content.title,
        styleName: "flex bg-white text-semesBlue py-2 px-4 rounded mx-1",
        icon: "src/assets/icons/delete.svg",
        handleClick: ()=>fetchChatroom(content.chatRoomId)
      };
  
      setChatroomComponents((prev) => [...prev, chatroomBtnProp]);
    })

    return chatroomList
  };

  useEffect(() => {
    // 채팅방 목록 조회
    fetchChatrooms();

  });

  const footStyle = "flex bg-transparent text-white py-2 px-4 rounded mx-1";

  const footerComponents: ButtonWithIconProps[] = [
    {
      btnName: "규정 확인하기",
      styleName: footStyle,
      icon: "/src/assets/icons/book-open-text.svg",
    },
    {
      btnName: "로그아웃",
      styleName: footStyle,
      icon: "/src/assets/icons/logout.svg",
    },
  ];

  const createChatRoom = (message: string) => {
    const result = {
      data: {
        chatroomId: 1,
        title: "채팅방 제목",
      },
    };

    setCurChatroomId(result.data.chatroomId);

    const chatroomBtnProp = {
      btnName: result.data.title,
      styleName: "flex bg-white text-semesBlue py-2 px-4 rounded mx-1",
      icon: "src/assets/icons/delete.svg",
    };

    setChatroomComponents((prev) => [...prev, chatroomBtnProp]);
  };

  const addInitialQnA = (message: string) => {
    const newQnA: QnA = {
      chatId: -1,
      question: new BaseMessage("question", message),
      answer: new BaseMessage("answer", ""),
      docs: [],
    };
    setQnAs((qnas) => [...qnas, newQnA]);
  };

  const sendMessage = async (message: string) => {
    if (isLoading || !message) return;

    setIsLoading(true);

    if (curChatroomId === -1) {
      createChatRoom(message);
    }

    // 초기 QnA 데이터 세팅
    addInitialQnA(message);
    
    // 출처 요청
    const docs = await searchDocsAPI(message);
    setQnAs((qnas) => {
      const updatedQnAs = [...qnas];
      updatedQnAs[qnas.length - 1].docs = docs;
      return updatedQnAs;
    });

    // 답변 생성
    handleGenerateResponse(message);
  };

  const handleGenerateResponse = async (message: string) => {
    try {
      const res = await generateAPI(qnas, message);
      if (!res.body) return;

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        let chunk = '';
        if (value instanceof Uint8Array) {
          chunk = decoder.decode(value, { stream: true });
        } else if (typeof value === 'string') {
          chunk = value;
        }

        // 청크가 있다면 추가
        if (chunk) {
          setQnAs((prevQnAs) => {
            const updatedQnAs = [...prevQnAs];
            const lastQnA = updatedQnAs[updatedQnAs.length - 1];
            lastQnA.answer.content += chunk;

            return updatedQnAs;
          });
        }
      }
    } catch (error) {
      console.error("Error:", error);
    } finally {
    
      // TODO : 채팅 저장하기
      const chatId = qnas[qnas.length - 1] ? qnas[qnas.length - 1].chatId + 1 : 1;

      // 채팅 id 및 답변 완료 여부 등록
      setQnAs((qnas) => {
        const updatedQnAs = [...qnas];
        const lastQnA = updatedQnAs[qnas.length - 1];
        lastQnA.isAnswered = true;
        lastQnA.chatId = chatId;
        return updatedQnAs;
      });

      setIsLoading(false);
    }
  };

  const handleFeedback = (qna: QnA, isPositive: boolean) => {
    setQnAs((qnas) => {
      const updatedQnAs = [...qnas];
      const updatedQnA = updatedQnAs.find(curQnA => curQnA.chatId === qna.chatId);
      if (updatedQnA) {
        updatedQnA.isPositive = isPositive;
      }
      return updatedQnAs;
    });
  };

  return (
    <SembotLayout
      hideHeader={true}
      isRule={false}
      sidebarComponents={chatroomComponents}
      footerComponents={footerComponents}
    >
      <div className="w-full h-full text-left">
        <ChatView
          qnas={qnas}
          onSendMessage={sendMessage}
          isLoading={isLoading}
          onFeedback={handleFeedback}
        />
      </div>
    </SembotLayout>
  );
};

export default Chat;

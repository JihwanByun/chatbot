import React, { useEffect, useRef, useState } from "react";
import InputWithIcon from "@components/atoms/input/InputWithIcon";
import { Message } from "@components/chat/ChatMessage";
import ChatMessage from "@components/chat/ChatMessage";
import ChatCategories, { ChatCategory } from "./ChatCategories";
import ChatDocs from "./ChatDocs";
import ButtonOnlyIcon from "@components/atoms/button/ButtonOnlyIcon";

export interface DocMetadata {
  source : string;
  page : number;
}

export interface Doc {
  metadata : DocMetadata;
  content : string;
}

export interface QnA {
  /** 채팅 id */
  chatId? : string
  /** 질문 */
  question : Message
  /** 답변 */
  answer : Message
  /** 출처 자료 목록 */
  docs : Doc[]
  /** 긍적적 답변인지 */
  isPositive? : boolean
  /** 답변 다 받았는지 */
  isAnswered? : boolean
}

export interface ChatViewProps {
  /** QnA 쌍 배열 */
  qnas: QnA[];
  /** 메시지 보냈을 때 실행되는 함수 */
  onSendMessage: (message: string) => void;
  /** 피드백 했을 때 실행되는 함수 */
  onFeedback: (qna : QnA, isPositive : boolean) => void;
  /** 로딩 상태 */
  isLoading: boolean;
  /** 카테고리 목록 */
  categories: ChatCategory[]
}

const ChatView: React.FC<ChatViewProps> = ({ qnas, onSendMessage, onFeedback, isLoading, categories }) => {
  const [initInputValue, setInitInputValue] = useState<string>("");
  
  const lastQnARef = useRef<QnA | null>(null);
  const chatContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // 마지막 QnA가 변경되었고, isAnswered나 content가 변경된 경우에만 스크롤
    const lastQnA = qnas[qnas.length - 1];
    const shouldScroll = lastQnA && (
      !lastQnARef.current || 
      lastQnA.answer.content !== lastQnARef.current.answer.content ||
      lastQnA.isAnswered !== lastQnARef.current.isAnswered
    );

    if (shouldScroll && chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }

    // 현재 마지막 QnA 상태 저장
    lastQnARef.current = lastQnA;
  }, [qnas]);
  
  const onClickCategory = (category: ChatCategory) => {
    setInitInputValue(category.prompt);
  };

  const renderFeedbackButton = (qna : QnA) => {
    if(qna.isPositive === true){
      return <ButtonOnlyIcon icon="src/assets/icons/like-fill.svg" styleName="hover:scale-110 transition-transform duration-200 ease-in-out" width={18} disabled={false}/>
    }else if(qna.isPositive === false){
      return <ButtonOnlyIcon icon="src/assets/icons/dislike-fill.svg" styleName="hover:scale-110 transition-transform duration-200 ease-in-out" width={18} disabled={false}/>
    }else{
      return <>
        <ButtonOnlyIcon icon="src/assets/icons/like.svg" styleName="hover:scale-110 transition-transform duration-200 ease-in-out" width={18} onClick={()=>onFeedback(qna,true)} disabled={false}/>
        <ButtonOnlyIcon icon="src/assets/icons/dislike.svg" styleName="hover:scale-110 transition-transform duration-200 ease-in-out" width={18} onClick={()=>onFeedback(qna,false)} disabled={false}/>
      </>
    }
  }

  return (
    <div className="mb-4 w-full h-full flex justify-center items-center text-gray-600">
      {qnas.length === 0 ? (
        <div className="w-full max-w-3xl h-full flex flex-col justify-center items-center px-4">
          <div className="flex justify-center items-center">
            <img src="/src/assets/icons/logo.svg" alt="SEMES" width="300rem" />
          </div>
          <InputWithIcon
            isLoading={isLoading}
            iconPath={"/src/assets/icons/send.svg"}
            onIconClick={onSendMessage}
            initInputValue={initInputValue}
          />
          <div className="mt-6">
            <ChatCategories categories={categories} onCategoryClick={onClickCategory} />
          </div>
        </div>
      ) : (
        <div className="relative w-full h-full flex flex-col justify-center items-center">
          <div ref={chatContainerRef} className="absolute top-0 w-full h-[88svh] overflow-auto flex justify-center pr-6 pl-2">
            <div className="w-full max-w-3xl h-full space-y-4 mt-6">
              {qnas.map((qna: QnA, index) => (
                <div key={index}>
                  <ChatMessage message={qna.question} />
                  <div className="my-6"></div>
                  <ChatMessage message={qna.answer} />
                  {qna.isAnswered &&
                    <>
                      <div className="ml-16">
                        <div className="my-6">
                          <ChatDocs docs={qna.docs} />
                        </div>
                        <div className="flex justify-start items-center my-4 space-x-2">
                          {renderFeedbackButton(qna)}
                        </div>
                      </div>
                      <hr className="ml-3 my-6"/>
                    </>
                  }
                </div>
              ))}
            </div>
          </div>
          <div className="absolute bottom-0 w-full max-w-3xl mb-10 px-4">
            <InputWithIcon
              isLoading={isLoading}
              iconPath={"/src/assets/icons/send.svg"}
              onIconClick={onSendMessage}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatView;

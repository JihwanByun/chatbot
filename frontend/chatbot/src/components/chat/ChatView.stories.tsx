import type { Meta } from '@storybook/react';
import ChatView, { QnA } from './ChatView';
import { BaseMessage } from './ChatMessage';
import { useState } from 'react';

const meta: Meta<typeof ChatView> = {
  title: 'COMPONENTS/CHAT/ChatView',
  component: ChatView,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
};
export default meta;

const sampleDocs = [
  {
    metadata: {
      source: "3_07_취업규칙(240214).pdf",
      page: 4
    },
    content: "입사다음연도1월1일에는다음각호의연차유급휴가를준다.<신설2021.12.28.>\n 1. 15일의연차유급휴가 ..."
  },
  {
    metadata: {
      source: "3_04_감사규칙(230605).pdf",
      page: 20
    },
    content: "[별지제8호]<개정2013.5.8>\n질문서\n제목:\n질문사항 답변사항\n홍길동귀하 홍길동귀하 ..."
  }
]

const sampleQnAs : QnA[] = [
  {
    chatId : "1234",
    question : new BaseMessage("question", "포탈 시스템 문의에 대해 질문이 있습니다."),
    answer : new BaseMessage("answer", "포탈 시스템 관련 정보를 제공해 드립니다."),
    docs : sampleDocs
  }
];

export const Default = () => {
  const [qnas, setQnAs] = useState<QnA[]>(sampleQnAs);
  const [isLoading, setIsLoading] = useState<boolean>(false); // 로딩 상태 추가

  const handleFeedback = (qna : QnA, isPositive: boolean) => {
    const updatedQnAs = [...qnas];
    const updatedQnA = updatedQnAs.find(
      curQnA => curQnA.chatId === qna.chatId
    );
    if(updatedQnA !== undefined){
      updatedQnA.isPositive = isPositive;
      console.log(isPositive);
    }
    return updatedQnAs;
  };

  const handleSendMessage = (message: string) => {
    setIsLoading(true); // 로딩 시작

    const defaultQnA : QnA = {
      question : new BaseMessage("question",message), 
      answer : new BaseMessage("answer",""),
      docs : []
    }

    // 초기 메시지 저장
    setQnAs((qnas)=>{
      return [...qnas, defaultQnA];
    });

    // 출처 검색
    const docs = sampleDocs;

    setQnAs((qnas)=>{
      const updatedQnAs = [...qnas];
      updatedQnAs[qnas.length-1].docs = docs;
      return updatedQnAs;
    });

    const answer = ["답변", " ", "입", "니", "다", ".", " ", "더", " ", "자", "세", "한", " ", "정", "보", "를", " ", "원", "하", "시", "나", "요", "?"];

    answer.forEach((chunk, index) => {
      setTimeout(() => {
        setQnAs((qnas) => {
          const updatedQnAs = [...qnas];
          const lastQnA = updatedQnAs[qnas.length-1];
          lastQnA.answer.content += chunk;

          if (index === answer.length - 1) {
            setIsLoading(false); // 로딩 종료
            lastQnA.isAnswered = true;
          }
          return updatedQnAs;
        });
      }, index * 100); // Delay each chunk by 100 ms
    });
  };

  return <ChatView qnas={qnas} onSendMessage={handleSendMessage} isLoading={isLoading} onFeedback={handleFeedback} />;
};
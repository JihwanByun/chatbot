import React, { useEffect, useRef, useState } from "react";
import InputWithIcon from "@components/atoms/input/InputWithIcon";
import { Message } from "@components/chat/ChatMessage";
import ChatMessage from "@components/chat/ChatMessage";
import ChatCategories, { ChatCategory } from "./ChatCategories";
import ChatDocs from "./ChatDocs";
import ButtonOnlyIcon from "@components/atoms/button/ButtonOnlyIcon";
import Swal from "sweetalert2";

export interface DocMetadata {
  source: string;
  page: number;
}

export interface Doc {
  metadata: DocMetadata;
  content: string;
}

export interface QnA {
  /** 채팅 id */
  chatId?: string;
  /** 질문 */
  question: Message;
  /** 답변 */
  answer: Message;
  /** 출처 자료 목록 */
  docs: Doc[];
  /** 긍적적 답변인지 */
  isPositive?: boolean;
  /** 답변 다 받았는지 */
  isAnswered?: boolean;
}

export interface ChatViewProps {
  /** QnA 쌍 배열 */
  qnas: QnA[];
  /** 메시지 보냈을 때 실행되는 함수 */
  onSendMessage: (message: string) => void;
  /** 피드백 했을 때 실행되는 함수 */
  onFeedback: (qna: QnA, isPositive: boolean, nagativeReason?: string) => void;
  /** 로딩 상태 */
  isLoading: boolean;
  /** 카테고리 목록 */
  categories: ChatCategory[];
  /** 부정 피드백 이유 목록 */
  feedbackReasons: string[];
}

const ChatView: React.FC<ChatViewProps> = ({
  qnas,
  onSendMessage,
  onFeedback,
  isLoading,
  categories,
  feedbackReasons,
}) => {
  const [initInputValue, setInitInputValue] = useState<string>("");

  const lastQnARef = useRef<QnA | null>(null);
  const chatContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // 마지막 QnA가 변경되었고, isAnswered나 content가 변경된 경우에만 스크롤
    const lastQnA = qnas[qnas.length - 1];
    const shouldScroll =
      lastQnA &&
      (!lastQnARef.current ||
        lastQnA.answer.content !== lastQnARef.current.answer.content ||
        lastQnA.isAnswered !== lastQnARef.current.isAnswered);

    if (shouldScroll && chatContainerRef.current) {
      chatContainerRef.current.scrollTop =
        chatContainerRef.current.scrollHeight;
    }

    // 현재 마지막 QnA 상태 저장
    lastQnARef.current = lastQnA;
  }, [qnas]);

  // 버튼 HTML 동적 생성
  const generateButtonsHtml = (reasons: string[]) => {
    return `
      <div class="flex flex-wrap gap-1.5 justify-center py-4">
        ${reasons
          .map(
            (reason) => `
          <button class="reason-btn bg-white border border-gray-300 hover:border-blue-500 text-gray-700 px-4 py-2 rounded-md text-sm">
            ${reason}
          </button>
        `
          )
          .join("")}
        <button class="reason-btn bg-white border border-gray-300 hover:border-blue-500 text-gray-700 px-4 py-2 rounded-md text-sm other-reason">
          기타
        </button>
      </div>
    `;
  };

  const feedbackHtml = generateButtonsHtml(feedbackReasons);

  const onClickCategory = (category: ChatCategory) => {
    setInitInputValue(category.prompt);
  };

  const feedbackNagative = async (qna: QnA) => {
    await Swal.fire({
      title: "더 자세히 알려주세요",
      html: feedbackHtml,
      showCancelButton: true,
      showConfirmButton: false,
      cancelButtonText: "닫기",
      customClass: {
        popup: "rounded-lg",
        cancelButton:
          "bg-white border border-gray-300 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-50",
      },
      width: "36rem", // 너비를 픽셀 단위로 설정
      didOpen: () => {
        // 버튼 클릭 이벤트 핸들러
        const buttons = document.querySelectorAll(".reason-btn");
        buttons.forEach((button) => {
          button.addEventListener("click", async function (e) {
            const target = e.target as HTMLButtonElement;

            if (target.classList.contains("other-reason")) {
              // 기타 이유 입력 모달
              const { value: otherReason } = await Swal.fire({
                title: "의견을 자세히 알려주세요",
                input: "text",
                inputPlaceholder: "의견을 입력해주세요",
                showCancelButton: true,
                confirmButtonText: "제출",
                cancelButtonText: "취소",
                customClass: {
                  confirmButton: "bg-blue-500 text-white px-4 py-2 rounded-lg",
                  cancelButton:
                    "bg-white border border-gray-300 text-gray-700 px-4 py-2 rounded-lg ml-2",
                },
              });

              if (otherReason) {
                onFeedback(qna, false, otherReason);
                Swal.close();
              }
            } else {
              // 일반 이유 선택
              const selectedReason = target.textContent?.trim() || "";
              onFeedback(qna, false, selectedReason);
              Swal.close();
            }
          });
        });
      },
    });
  };

  const renderFeedbackButton = (qna: QnA) => {
    if (qna.isPositive === true) {
      return (
        <ButtonOnlyIcon
          icon="src/assets/icons/like-fill.svg"
          styleName="hover:scale-110 transition-transform duration-200 ease-in-out"
          width={18}
          disabled={false}
        />
      );
    } else if (qna.isPositive === false) {
      return (
        <ButtonOnlyIcon
          icon="src/assets/icons/dislike-fill.svg"
          styleName="hover:scale-110 transition-transform duration-200 ease-in-out"
          width={18}
          disabled={false}
        />
      );
    } else {
      return (
        <>
          <ButtonOnlyIcon
            icon="src/assets/icons/like.svg"
            styleName="hover:scale-110 transition-transform duration-200 ease-in-out"
            width={18}
            onClick={() => onFeedback(qna, true)}
            disabled={false}
          />
          <ButtonOnlyIcon
            icon="src/assets/icons/dislike.svg"
            styleName="hover:scale-110 transition-transform duration-200 ease-in-out"
            width={18}
            onClick={() => feedbackNagative(qna)}
            disabled={false}
          />
        </>
      );
    }
  };

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
            <ChatCategories
              categories={categories}
              onCategoryClick={onClickCategory}
            />
          </div>
        </div>
      ) : (
        <div className="relative w-full h-full flex flex-col justify-center items-center">
          <div
            ref={chatContainerRef}
            className="absolute top-0 w-full h-[88svh] overflow-auto flex justify-center pr-6 pl-2"
          >
            <div className="w-full max-w-3xl h-full space-y-4 mt-6">
              {qnas.map((qna: QnA, index) => (
                <div key={index}>
                  <ChatMessage message={qna.question} />
                  <div className="my-6"></div>
                  <ChatMessage message={qna.answer} />
                  {qna.isAnswered && (
                    <>
                      <div className="ml-16">
                        <div className="my-6">
                          <ChatDocs docs={qna.docs} />
                        </div>
                        <div className="flex justify-start items-center my-4 space-x-2">
                          {renderFeedbackButton(qna)}
                        </div>
                      </div>
                      <hr className="ml-3 my-6" />
                    </>
                  )}
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

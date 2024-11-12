import React, { useCallback, useEffect, useState } from "react";
import ChatView, { Doc, QnA } from "@components/chat/ChatView";
import ButtonWithIcon from "@components/atoms/button/ButtonWithIcon";
import { BaseMessage } from "@components/chat/ChatMessage";
import SembotLayout from "@pages/SembotLayout";
import Swal from "sweetalert2";
import { useNavigate } from "react-router-dom";
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
	createChatFeedbackAPI,
	deleteChatroomAPI,
} from "@apis/chat/chatApi";
import { logout } from "@apis/chat/userApi";
import { ChatCategory } from "@components/chat/ChatCategories";
import { getCategoryListAPI } from "@apis/category/categoryApi";
import { getFeedbackReasonList } from "@apis/feedback/feedbackApi";

export type ButtonWithIconProps = React.ComponentProps<typeof ButtonWithIcon>;

export type ChatroomButtonProps = ButtonWithIconProps & {
	chatroomId: number; // 또는 number (chatroomId의 타입에 따라)
};

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
	const [categories, setCategories] = useState<ChatCategory[]>([]);
	const [chatroomComponents, setChatroomComponents] = useState<
		ChatroomButtonProps[]
	>([]);
	const [feedbackReasons, setFeedbackReasons] = useState<string[]>([]);
	// // 피드백 내용들을 배열로 관리
	// const feedbackReasons = [
	//   "코드가 틀렸습니다",
	//   "데모코드를 사용해선 안 됐습니다",
	//   "스타일이 마음에 들지 않습니다",
	//   "올바른 사례가 아닌 말을 했습니다",
	//   "지시한 내용을 다 따르지 않았습니다",
	//   "기타"
	// ];

	const navigate = useNavigate();

	// UI Constants
	const footStyle =
		"flex bg-transparent text-white py-2 px-4 rounded mx-1 transform hover:translate-x-1 transition-all duration-200 cursor-pointer";
	const newChatButtonStyle =
		"flex bg-transparent border border-white text-white py-2 px-4 rounded mx-1 hover:bg-blue-900 transition-colors duration-100 ease-in-out";
	const chatroomButtonStyle =
		"flex bg-white text-semesBlue py-2 px-4 rounded mx-1 hover:bg-gray-200 transition-colors duration-100 ease-in-out relative";
	const deleteIconStyle =
		"hover:scale-125 transform-gpu origin-center transition-transform duration-200 absolute right-1 top-2.5";

	// Utility Functions
	const createChatroomButton = (
		chatroomId: number,
		btnName: string,
		style: string,
		icon: string,
		handleClick?: () => void,
		iconStyleName?: string,
		handleIconClick?: () => void
	): ChatroomButtonProps => ({
		btnName,
		styleName: style,
		icon,
		handleClick,
		iconStyleName,
		handleIconClick,
		chatroomId,
	});

	// Chat Room Management
	const newChatProp = createChatroomButton(
		-1,
		"새채팅",
		newChatButtonStyle,
		"/src/assets/icons/plus.svg",
		() => {
			setState((prev) => ({ ...prev, curChatroomId: -1, qnas: [] }));
		}
	);

	const footerComponents: ButtonWithIconProps[] = [
		createChatroomButton(
			-1,
			"규정 확인하기",
			footStyle,
			"/src/assets/icons/book-open-text.svg",
			() => navigate("/board")
		),
		createChatroomButton(
			-1,
			"로그아웃",
			footStyle,
			"/src/assets/icons/logout.svg",
			() => {
				logout();
				navigate("/");
			}
		),
	];

	// API Handlers
	const handleApiError = useCallback(
		(error: unknown, customMessage: string) => {
			console.error("Error:", error);
			setState((prev) => ({
				...prev,
				error: error instanceof Error ? error.message : customMessage,
			}));
		},
		[]
	);

	const deleteChatroom = async (chatroomId: number) => {
		await deleteChatroomAPI(chatroomId);
		setChatroomComponents((prev) =>
			prev.filter((component) => component.chatroomId !== chatroomId)
		);
		setState((prev) => ({ ...prev, curChatroomId: -1, qnas: [] }));
	};

	const createChatroomBtnProps = (content: ChatroomResponse) => {
		return createChatroomButton(
			content.chatRoomId,
			content.title,
			chatroomButtonStyle,
			"src/assets/icons/delete.svg",
			() => fetchChatroom(content.chatRoomId),
			deleteIconStyle,
			() => {
				Swal.fire({
					title: "삭제하시겠습니까?",
					text: "삭제 후 복구는 불가능합니다.",
					icon: "warning",
					showCancelButton: true,
					// confirmButtonColor: '#d33',
					// cancelButtonColor: '#3085d6',
					confirmButtonText: "삭제",
					cancelButtonText: "취소",
					customClass: {
						confirmButton:
							"bg-red-500 text-white px-4 py-2 rounded-lg focus:ring-0 focus:outline-none active:bg-red-500 hover:bg-red-600",
						cancelButton:
							"bg-white border-2 border-gray-300 text-gray-700 px-4 py-2 rounded-lg ml-2 hover:bg-gray-200",
					},
					buttonsStyling: false, // 기본 스타일링 비활성화
				}).then((result) => {
					if (result.isConfirmed) {
						deleteChatroom(content.chatRoomId);
						Swal.fire("삭제되었습니다.");
					}
				});
			}
		);
	};

	const fetchChatroom = useCallback(
		async (chatroomId: number) => {
			try {
				const chatroom: ChatroomDetail | null = await getChatroomDetailAPI(
					chatroomId
				);
				if (!chatroom)
					throw new Error("채팅방 정보를 가져오는데 실패했습니다.");

				setState((prev) => ({
					...prev,
					curChatroomId: chatroomId,
					qnas: chatroom.qnas,
					error: null,
				}));
			} catch (error) {
				handleApiError(error, "채팅방 정보를 가져오는데 실패했습니다.");
			}
		},
		[handleApiError]
	);

	const fetchChatrooms = useCallback(async () => {
		if (state.isFetchingChatrooms || !state.hasNextChatroom || state.error)
			return;

		try {
			setState((prev) => ({ ...prev, isFetchingChatrooms: true, error: null }));

			const chatroomList: ChatroomList | null = await getChatroomListAPI(
				state.currentChatroomPage
			);
			if (!chatroomList)
				throw new Error("채팅방 목록을 가져오는데 실패했습니다.");

			const newComponents = chatroomList.contents.map(createChatroomBtnProps);

			setState((prev) => ({
				...prev,
				hasNextChatroom: chatroomList.hasNext,
				currentChatroomPage: prev.currentChatroomPage + 1,
			}));

			setChatroomComponents((prev) => [...prev, ...newComponents]);
		} catch (error) {
			handleApiError(error, "채팅방 목록을 가져오는데 실패했습니다.");
		} finally {
			setState((prev) => ({ ...prev, isFetchingChatrooms: false }));
		}
	}, [
		state.isFetchingChatrooms,
		state.hasNextChatroom,
		state.error,
		state.currentChatroomPage,
		handleApiError,
		fetchChatroom,
	]);

	// Message Handling
	const createQnA = (message: string, docs: Doc[]): QnA => ({
		question: new BaseMessage("question", message),
		answer: new BaseMessage("answer", ""),
		docs,
		isAnswered: false,
	});

	const handleStreamResponse = useCallback(
		async (
			reader: ReadableStreamDefaultReader<Uint8Array>,
			decoder: TextDecoder
		): Promise<string> => {
			let fullAnswer = "";

			while (true) {
				const { done, value } = await reader.read();
				if (done) break;

				const chunk =
					value instanceof Uint8Array
						? decoder.decode(value, { stream: true })
						: (value as string);
				if (chunk) {
					fullAnswer += chunk;
					setState((prev) => ({
						...prev,
						qnas: prev.qnas.map((qna, idx) =>
							idx === prev.qnas.length - 1
								? { ...qna, answer: new BaseMessage("answer", fullAnswer) }
								: qna
						),
					}));
				}
			}

			return fullAnswer;
		},
		[]
	);

	const handleGenerateResponse = useCallback(
		async (message: string, activeChatroomId: number, currentQnA: QnA) => {
			try {
				const res = await generateAPI(state.qnas, message);
				if (!res.body) throw new Error("응답 생성에 실패했습니다.");

				const fullAnswer = await handleStreamResponse(
					res.body.getReader(),
					new TextDecoder("utf-8")
				);

				const updatedQnA = {
					...currentQnA,
					answer: new BaseMessage("answer", fullAnswer),
				};

				const chat = await createChatAPI(activeChatroomId, updatedQnA);
				if (!chat) throw new Error("채팅 저장에 실패했습니다.");

				setState((prev) => ({
					...prev,
					qnas: prev.qnas.map((qna, idx) =>
						idx === prev.qnas.length - 1
							? {
									...qna,
									docs: updatedQnA.docs,
									chatId: chat.chatId,
									isAnswered: true,
							  }
							: qna
					),
				}));
			} catch (error) {
				handleApiError(error, "응답 생성에 실패했습니다.");
			}
		},
		[state.qnas, handleStreamResponse, handleApiError]
	);

	const sendMessage = useCallback(
		async (message: string) => {
			if (state.isLoading || !message) return;

			try {
				setState((prev) => ({ ...prev, isLoading: true, error: null }));

				let activeChatroomId = state.curChatroomId;
				if (activeChatroomId === -1) {
					const result = await createChatroomAPI(message);
					if (!result) throw new Error("채팅방 생성에 실패했습니다.");

					activeChatroomId = result.chatRoomId;
					setState((prev) => ({ ...prev, curChatroomId: activeChatroomId }));

					const chatroomButton = createChatroomBtnProps(result);

					setChatroomComponents((prev) => [
						newChatProp,
						chatroomButton,
						...prev.slice(1),
					]);
				}
				const newQnA = createQnA(message, []);

				setState((prev) => ({
					...prev,
					qnas: [...prev.qnas, newQnA],
				}));

				const docs = await searchDocsAPI(message);

				newQnA.docs = docs;

				await handleGenerateResponse(message, activeChatroomId, newQnA);
			} catch (error) {
				handleApiError(error, "메시지 전송에 실패했습니다.");
			} finally {
				setState((prev) => ({ ...prev, isLoading: false }));
			}
		},
		[
			state.isLoading,
			state.curChatroomId,
			handleGenerateResponse,
			handleApiError,
			fetchChatroom,
		]
	);

	const handleFeedback = useCallback(
		async (qna: QnA, isPositive: boolean, negativeReason?: string) => {
			if (!qna.chatId) return;

			// 긍정 피드백
			createChatFeedbackAPI(qna.chatId, isPositive, negativeReason);
			setState((prev) => ({
				...prev,
				qnas: prev.qnas.map((q) =>
					q.chatId === qna.chatId ? { ...q, isPositive } : q
				),
			}));
		},
		[]
	);

	const fetchCategories = async () => {
		const categories = await getCategoryListAPI();

		setCategories(categories);
	};

	const fetchFeedbackReasons = async () => {
		const feedbackReasons = await getFeedbackReasonList();
		setFeedbackReasons(feedbackReasons);
	};

	// Initial Load
	useEffect(() => {
		if (state.currentChatroomPage === 0) {
			fetchChatrooms();
		}
	}, [state.currentChatroomPage, fetchChatrooms]);

	useEffect(() => {
		fetchCategories();
		fetchFeedbackReasons();
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
					categories={categories}
					feedbackReasons={feedbackReasons}
				/>
			</div>
		</SembotLayout>
	);
};

export default Chat;

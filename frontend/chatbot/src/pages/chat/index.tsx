import React, { useCallback, useEffect, useState } from "react";
import ChatView, { Doc, QnA } from "@components/chat/ChatView";
import ButtonWithIcon from "@components/atoms/button/ButtonWithIcon";
import { BaseMessage } from "@components/chat/ChatMessage";
import SembotLayout from "@pages/SembotLayout";
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
import { ChatCategory } from "@components/chat/ChatCategories";
import { getChatCategoryListAPI as getCategoryListAPI } from "@apis/category/categoryApi";
import { getFeedbackReasonList } from "@apis/feedback/feedbackApi";
import { useAppDispatch, useAppSelector } from "@app/hooks";
import { logoutUser } from "@app/slices/userSlice";
import { deleteAlert, errorAlert } from "@util/alert";

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

	const dispatch = useAppDispatch();
	const user = useAppSelector((state) => state.users);	

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

	const handleError = (error : unknown) => {
		if (error instanceof Error) {
			errorAlert(error);
			setState((prev) => ({ ...prev, error: error.message}))
		}else{
			setState((prev) => ({ ...prev, error: "알 수 없는 오류가 발생했습니다."}))
		}
	}

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
			"/src/assets/icons/book-open-text-footer.svg",
			() => navigate("/board")
		),
		createChatroomButton(
			-1,
			"로그아웃",
			footStyle,
			"/src/assets/icons/logout.svg",
			async () => {  // async 추가
				try {
					await dispatch(logoutUser());  // logout 완료 대기
					navigate("/");  // 로그아웃 완료 후 navigate
				} catch (error) {
					console.log(error);
					errorAlert(new Error("로그아웃에 실패하였습니다."));
				}
			}
		),
	];

	const deleteChatroom = async (chatroomId: number) => {
		await deleteChatroomAPI(chatroomId);
		setChatroomComponents((prev) =>
			prev.filter((component) => component.chatroomId !== chatroomId)
		);
		setState((prev) => ({ ...prev, curChatroomId: -1, qnas: [] }));
	};

	const fetchChatroom = useCallback(
		async (chatroomId: number) => {
			const chatroom: ChatroomDetail = await getChatroomDetailAPI(
				chatroomId
			);

			setState((prev) => ({
				...prev,
				curChatroomId: chatroomId,
				qnas: chatroom.qnas,
				error: null,
			}));
		},
		[]
	);

	const createChatroomBtnProps = useCallback((content: ChatroomResponse) => {
		return createChatroomButton(
			content.chatRoomId,
			content.title,
			chatroomButtonStyle,
			"src/assets/icons/delete.svg",
			() => fetchChatroom(content.chatRoomId),
			deleteIconStyle,
			()=>deleteAlert(()=>deleteChatroom(content.chatRoomId))
		);
	},[fetchChatroom]);

	const fetchChatrooms = useCallback(async () => {
		if (state.isFetchingChatrooms || !state.hasNextChatroom || state.error)
			return;

		try {
			setState((prev) => ({ ...prev, isFetchingChatrooms: true, error: null }));

			const chatroomList: ChatroomList = await getChatroomListAPI(
				state.currentChatroomPage
			);

			const newComponents = chatroomList.contents.map(createChatroomBtnProps);

			setState((prev) => ({
				...prev,
				hasNextChatroom: chatroomList.hasNext,
				currentChatroomPage: prev.currentChatroomPage + 1,
			}));

			setChatroomComponents((prev) => [...prev, ...newComponents]);
		} catch (error) {
			handleError(error);
		} finally {
			setState((prev) => ({ ...prev, isFetchingChatrooms: false }));
		}
	}, [
		state.isFetchingChatrooms,
		state.hasNextChatroom,
		state.error,
		state.currentChatroomPage,
		createChatroomBtnProps
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
		async (message: string, currentQnA: QnA) => {
			const res = await generateAPI(state.qnas, message, user.level);
			if (!res.body) throw new Error("응답 생성에 실패했습니다.");

			const fullAnswer = await handleStreamResponse(
				res.body.getReader(),
				new TextDecoder("utf-8")
			);

			let activeChatroomId = state.curChatroomId;
			if (activeChatroomId === -1) {
				const result = await createChatroomAPI(message);

				activeChatroomId = result.chatRoomId;
				setState((prev) => ({ ...prev, curChatroomId: activeChatroomId }));

				const chatroomButton = createChatroomBtnProps(result);

				setChatroomComponents((prev) => [
					newChatProp,
					chatroomButton,
					...prev.slice(1),
				]);
			}

			const updatedQnA = {
				...currentQnA,
				answer: new BaseMessage("answer", fullAnswer),
			};

			const chat = await createChatAPI(activeChatroomId, updatedQnA);

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
		},
		[state.qnas, state.curChatroomId, handleStreamResponse, createChatroomBtnProps, newChatProp, user.level]
	);

	const sendMessage = useCallback(
		async (message: string) => {
			if (state.isLoading || !message) return;

			try {
				setState((prev) => ({ ...prev, isLoading: true, error: null }));

				const docs = await searchDocsAPI(message, user.level);

				const newQnA = createQnA(message, docs);

				setState((prev) => ({
					...prev,
					qnas: [...prev.qnas, newQnA],
				}));

				await handleGenerateResponse(message, newQnA);
			} catch (error) {
				handleError(error);
			} finally {
				setState((prev) => ({ ...prev, isLoading: false }));
			}
		},
		[
			state.isLoading,
			handleGenerateResponse,
			user.level,
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
		if (!state.error && state.currentChatroomPage === 0) {
			fetchChatrooms();
		}
	}, [state.error, state.currentChatroomPage, fetchChatrooms]);

	useEffect(() => {
		console.log(user);
		if(!state.error){
			fetchCategories();
			fetchFeedbackReasons();
			setChatroomComponents([newChatProp]);
		}
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

import React, { useState, useEffect } from "react";
import Dropdown from "@components/atoms/dropdown/Dropdown";
import FeedbackCardWithHashtag from "@components/atoms/card/FeedbackCardWithHashtag";
import Paging from "@components/atoms/paging/Paging";
import { useAppDispatch, useAppSelector } from "@app/hooks";
import { fetchFeedbacksByPage } from "@app/slices/feedbackSlice";

const items = ["긍정", "부정"];

const FeedbackManagement = () => {
	const dispatch = useAppDispatch();
	const { feedbacks, loading } = useAppSelector((state) => state.feedbacks);
	const [page, setPage] = useState(1);
	const [size, setSize] = useState(6); // 초기값은 6으로 설정
	const [sortBy, setSortBy] = useState("createdAt");
	const [sortDir, setSortDir] = useState("desc");
	const [isPositive, setIsPositive] = useState<null | boolean>(null);

	useEffect(() => {
		dispatch(
			fetchFeedbacksByPage({
				isPositive,
				page,
				size,
				sortBy,
				sortDir,
			})
		);
	}, [dispatch, isPositive, page, size, sortBy, sortDir]);

	// setState가 비동기적으로 작동하기 때문에
	// setState가 setState를 호출한 직후에는 상태가 즉시 업데이트되지 않고,
	// 컴포넌트가 다음 렌더링 사이클에서 업데이트됩니다.

	const handleDropdown = (item: string) => {
		if (item == "긍정") {
			setPage(1);
			setIsPositive(true);
		} else if (item == "부정") {
			setPage(1);
			setIsPositive(false);
		}
	};

	// 뷰포트 높이에 따라 cardsPerPage를 계산하는 함수
	const calculateCardsPerPage = () => {
		const viewportHeight = window.innerHeight;

		if (viewportHeight < 800) return 6;
		else if (viewportHeight < 1200) return 9;
		else return 12;
	};

	useEffect(() => {
		// 페이지 로드 시와 화면 크기 변경 시 cardsPerPage 업데이트
		const updateCardsPerPage = () => setSize(calculateCardsPerPage());
		updateCardsPerPage();

		window.addEventListener("resize", updateCardsPerPage);
		return () => window.removeEventListener("resize", updateCardsPerPage);
	}, []);

	// 카드 목록 생성
	const getHashTags = () => {
		const Hashtags = [];

		// 실제 데이터로 카드 생성
		feedbacks.contents.forEach((element, index) => {
			const hashTags = [];
			hashTags.push(element.isPositive ? "긍정적" : "부정적");
			if (element.negativeReason) {
				// TODO : 여기에 요약이 들어가면 좋을것같습니다.
				hashTags.push(element.negativeReason.substring(0, 3));
			}
			Hashtags.push(
				<div className="w-[25%] m-5" key={element.chatId || index}>
					<FeedbackCardWithHashtag
						question={element.question.substring(0, 20)}
						answer={element.answer.substring(0, 20)}
						hashtags={hashTags}
						className="relative flex w-full m-5 flex-col rounded-xl bg-white bg-clip-border text-gray-700 shadow-md"
					/>
				</div>
			);
		});

		if (Hashtags.length % 3 != 0) {
			for (let i = 0; i < Hashtags.length % 3; i++) {
				Hashtags.push(
					<div className="w-[25%] m-5">
						<div className="relative flex w-full m-5 flex-col rounded-xl bg-white bg-clip-border text-gray-700 shadow-md"></div>
					</div>
				);
			}
		}
		return Hashtags;
	};

	return (
		<div>
			<div className="flex justify-end">
				<Dropdown
					items={items}
					width="7rem"
					className={"right-[10%]"}
					onSelect={handleDropdown}
				/>
			</div>
			<div className="flex flex-row flex-wrap justify-center">
				{getHashTags()}
			</div>
			<div className="flex justify-center">
				<div className="absolute bottom-5 mt-4 mb-[8%]">
					<Paging
						curPage={page}
						onPageChange={setPage}
						totalPage={feedbacks.totalPages} // 전체 페이지 수 계산
					/>
				</div>
			</div>
		</div>
	);
};

export default FeedbackManagement;

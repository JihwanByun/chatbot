import React, { useState, useEffect } from "react";
import TableWithIconAndButton from "@components/atoms/table/TableWithIcon";
import Paging from "@components/atoms/paging/Paging";
import InputSearch from "@components/atoms/input/InputSearch";
import { useAppDispatch, useAppSelector } from "@app/hooks";
import { fetchMembersByPage } from "@app/slices/memberSlice";

const header = ["", "사번", "이름", "부서", "쓰기 권한", "회원레벨"];
const columns = ["ssafy_1", "김광현", "개발부", "O", "3"];

const MemberManagement = () => {
	const dispatch = useAppDispatch();
	const { members, loading } = useAppSelector((state) => state.members);
	const [page, setPage] = useState(1);
	const [size, setSize] = useState(10); // 초기값 설정
	const [sortBy, setSortBy] = useState("createdAt");
	const [sortDir, setSortDir] = useState("desc");

	// 일단 출력까지만 해봅시다.
	// useEffect(() => {

	// 	if(loading){
	// 		console.log("데이터 로딩 중..")
	// 	} else{
	// 		console.log("데이터 로딩 완료",members);
	// 	}

	// 	dispatch(
	// 		fetchMembersByPage({
	// 			page,
	// 			size,
	// 			sortBy,
	// 			sortDir,
	// 		})
	// 	);
	// }, [dispatch, page, size, sortBy, sortDir]);

	// useEffect(() => {

	// 	console.log(members);
	// }, [members]);

	const iconPaths = [];
	for (let i = 1; i <= 13; i++) {
		iconPaths.push("/src/assets/icons/user-profile-ex.svg");
	}

	// 뷰포트 높이에 따라 itemsPerPage를 계산하는 함수
	const calculateItemsPerPage = () => {
		const viewportHeight = window.innerHeight;

		if (viewportHeight < 800) return 10;
		else if (viewportHeight < 1200) return 9;
		else return 12;
	};

	useEffect(() => {
		// 페이지 로드 시와 화면 크기 변경 시 itemsPerPage 업데이트
		const updateItemsPerPage = () => setSize(calculateItemsPerPage());
		updateItemsPerPage();

		window.addEventListener("resize", updateItemsPerPage);
		return () => window.removeEventListener("resize", updateItemsPerPage);
	}, []);

	// members 데이터를 페칭하기 위한 useEffect
	useEffect(() => {
		// console.log("여기")
		dispatch(
			fetchMembersByPage({
				page,
				size,
				sortBy,
				sortDir,
			})
		);
	}, [dispatch, page, size, sortBy, sortDir]);

	useEffect(() => {
		if (!loading ) {
			console.log("데이터 로딩 완료:", members);
		}
		currentItems();
	}, [loading, members]);

	// 테이블에 표시할 행 데이터 생성
	const getTableRowData = () => {
		console.log(members.contents)
		console.log(
		members?.contents?.map((member, index) => ({
			id: index + 1,
			columns: [
				member.employeeNum,
				member.name,
				member.department,
				member.role,
				member.level.toString(),
			],
		})) || []
	)
		
		return (
			members?.contents?.map((member, index) => ({
				id: index + 1,
				columns: [
					member.employeeNum,
					member.name,
					member.department,
					member.role,
					member.level.toString(),
				],
			})) || []
		);
	};

	// 현재 페이지에 맞는 항목들을 필터링
	const currentItems = () => { return getTableRowData().slice(size * (page - 1), size * page);}

	const handleClick = () => {};
		// 로딩이 완료된 후에만 테이블 렌더링
	if (loading) {
		return <div>로딩 중...</div>;
	}

	return (
		<div>
			<div>
				<InputSearch />
			</div>
			<div>
				<TableWithIconAndButton
					columns={header}
					rows={currentItems()}
					iconPaths={iconPaths}
					onIconClick={handleClick}
					width="50px"
				/>
			</div>
			<div className="flex justify-center">
				<div className="absolute bottom-5 mt-4 mb-[8%]">
					<Paging
						curPage={page}
						onPageChange={setPage}
						totalPage={members.totalPages} // 전체 페이지 수 계산
					/>
				</div>
			</div>
		</div>
	);
};

export default MemberManagement;

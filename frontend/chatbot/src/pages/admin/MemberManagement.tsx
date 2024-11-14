import React, { useState, useEffect } from "react";
import TableWithAvatarAndButton from "@components/atoms/table/TableWithAvatar";
import Paging from "@components/atoms/paging/Paging";
import InputSearch from "@components/atoms/input/InputSearch";
import { useAppDispatch, useAppSelector } from "@app/hooks";
import { fetchMembersByPage } from "@app/slices/memberSlice";
import { loginUser, updateUser } from "@app/slices/userSlice";
import ButtonPrimary from "@components/atoms/button/ButtonPrimary";
import Modal from "@pages/admin/Modal";
import Dropdown from "@components/atoms/dropdown/Dropdown";

const header = ["", "사번", "이름", "부서", "쓰기권한", "회원레벨", "정보변경"];
const columns = ["ssafy_1", "김광현", "개발부", "O", "3"];

// TableRowData 타입 정의
interface TableRowData {
	id: number;
	columns: (string | JSX.Element)[];
}

const MemberManagement = () => {
	const dispatch = useAppDispatch();
	const { members, loading } = useAppSelector((state) => state.members);
	const [page, setPage] = useState(1);
	const [size, setSize] = useState(10); // 초기값 설정
	const [sortBy, setSortBy] = useState("createdAt");
	const [sortDir, setSortDir] = useState("desc");
	const [tableData, setTableData] = useState<TableRowData[]>([]);
	const [isModalOpen, setIsModalOpen] = useState(false);
	const [selectedMemberId, setSelectedMemberId] = useState(null);
	const [newLevel, setNewLevel] = useState("");
	const [newRole, setNewRole] = useState("");

	useEffect(() => {
		if (loading) {
			console.log("데이터 로딩 중..");
		} else {
			console.log("데이터 로딩 완료", members);
		}

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
		if (members && members.contents) {
			const getTableRowData2 = () => {
				const TableRowDataList: TableRowData[] = [];
				members.contents.forEach((element, index) => {
					TableRowDataList.push({
						id: index + 1,
						columns: [
							element.employeeNum,
							element.name,
							element.department,
							element.role,
							String(element.level),
							<ButtonPrimary
								btnName="변경"
								styleName="bg-blue-200 hover:bg-blue-300 transition-colors duration-150"
								handleClick={() => openModal(element)}
							/>,
						],
					});
				});
				return TableRowDataList;
			};

			const currentItems = () => {
				// members.contents가 비어있을 때 빈 배열을 반환하도록 추가
				return members.contents.length > 0
					? getTableRowData2().slice(size * (page - 1), size * page)
					: [];
			};

			setTableData(currentItems());
		}
	}, [members, page, size]);

	// 뷰포트 높이에 따라 itemsPerPage를 계산하는 함수
	const calculateItemsPerPage = () => {
		const viewportHeight = window.innerHeight;

		if (viewportHeight < 800) return 10;
		else if (viewportHeight < 1200) return 9;
		else return 12;
	};

	const openModal = (member) => {
		setSelectedMemberId(member.userId);
		setNewLevel(String(member.level));
		setNewRole(member.role);
		setIsModalOpen(true);
	};

	const closeModal = () => {
		setIsModalOpen(false);
		setSelectedMemberId(null);
		setNewLevel("");
		setNewRole("");
	};

	const handleSaveChanges = () => {
		dispatch(
			updateUser({
				userId: selectedMemberId,
				level: Number(newLevel),
				role: newRole,
			})
		);

		console.log("변경된 level:", newLevel);
		console.log("변경된 role:", newRole);
		closeModal();
	};

	useEffect(() => {
		// 페이지 로드 시와 화면 크기 변경 시 itemsPerPage 업데이트
		const updateItemsPerPage = () => setSize(calculateItemsPerPage());
		updateItemsPerPage();

		window.addEventListener("resize", updateItemsPerPage);
		return () => window.removeEventListener("resize", updateItemsPerPage);
	}, []);

	const handleClick = () => { };
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
				<TableWithAvatarAndButton
					columns={header}
					rows={tableData}
					iconPaths={"/src/assets/icons/user_2.svg"}
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
			{isModalOpen && (
				<Modal onClose={closeModal}>
					<div className="p-4">
						<h2>정보 변경</h2>
						<div className="mt-4">
							<label>회원 레벨</label>
							<Dropdown
								buttonLabel={newLevel}
								items={["1", "2", "3"]}
								onSelect={setNewLevel}
							/>
						</div>
						<div className="mt-4">
							<label>쓰기 권한</label>
							<Dropdown
								buttonLabel={newRole}
								items={["USER", "USER_WRITE", "ADMIN"]}
								onSelect={setNewRole}
							/>
						</div>
						<div className="flex justify-end mt-6">
							<ButtonPrimary
								btnName="변경 완료"
								handleClick={handleSaveChanges}
							/>
						</div>
					</div>
				</Modal>
			)}
		</div>
	);
};

export default MemberManagement;

import React, { useState, useEffect } from "react";
import { useAppDispatch, useAppSelector } from "@app/hooks";
import TableWithAvatarAndButton from "@components/atoms/table/TableWithAvatar";
import Paging from "@components/atoms/paging/Paging";
import InputSearch from "@components/atoms/input/InputSearch";
import Modal from "@components/atoms/modal/Modal";
import DropdownBase from "@components/atoms/dropdown/DropdownBase";
import ButtonPrimary from "@components/atoms/button/ButtonPrimary";
import { fetchMembersByPage } from "@app/slices/memberSlice";
import { updateUser } from "@app/slices/userSlice";

interface TableRowData {
  id: number;
  columns: (string | JSX.Element)[];
}

interface SelectedMember {
  userId: number;
  level: number;
  role: string;
}

const searchTypeMapping = {
  이름: "name",
  사번: "employeeNum",
  부서: "department",
  권한: "role",
  레벨: "level",
};

const MemberManagement = () => {
  const dispatch = useAppDispatch();
  const { members, loading } = useAppSelector((state) => state.members);

  // State management
  const [page, setPage] = useState(1);
  const [size, setSize] = useState(10);
  const [sortBy, setSortBy] = useState("createdAt");
  const [sortDir, setSortDir] = useState("desc");
  const [searchType, setSearchType] = useState("name");
  const [searchValue, setSearchValue] = useState("");
  const [selectedLevels, setSelectedLevels] = useState<number[]>([]);

  // Modal state
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedMember, setSelectedMember] = useState<SelectedMember | null>(null);
  const [newLevel, setNewLevel] = useState("");
  const [newRole, setNewRole] = useState("");

  const header = ["", "사번", "이름", "부서", "권한", "레벨", "정보변경"];
  const levelOptions = [
    { id: 1, label: "Level 1" },
    { id: 2, label: "Level 2" },
    { id: 3, label: "Level 3" },
  ];
  const roleOptions = ["USER", "USER_WRITE", "ADMIN"];

  const handleSearch = (type: string, value: string) => {
    const mappedType = searchTypeMapping[type as keyof typeof searchTypeMapping] || "name";
    setSearchType(mappedType);
    setSearchValue(value);
    setPage(1); // 검색 시 첫 페이지로 이동
  };

  const handleLevelChange = (level: number) => {
    setSelectedLevels((prev) => {
      if (prev.includes(level)) {
        return prev.filter((l) => l !== level);
      }
      return [...prev, level];
    });
    setPage(1);
  };

  const openModal = (member: any) => {
    setSelectedMember({
      userId: member.userId,
      level: member.level,
      role: member.role,
    });
    setNewLevel(String(member.level));
    setNewRole(member.role);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedMember(null);
    setNewLevel("");
    setNewRole("");
  };

  const handleSaveChanges = async () => {
    if (selectedMember) {
      try {
        await dispatch(
          updateUser({
            userId: selectedMember.userId,
            level: Number(newLevel),
            role: newRole,
          })
        ).unwrap();

        await dispatch(
          fetchMembersByPage({
            page,
            size,
            sortBy,
            sortDir,
            [searchType]: searchValue,
          })
        );

        closeModal();
      } catch (error) {
        console.error("Failed to update user:", error);
      }
    }
  };

  const searchFields = ["name", "employeeNum", "department", "role", "level"];

  const getTableData = () => {
    if (!members.contents) return [];

    // 레벨 필터만 클라이언트에서 처리 (또는 이것도 서버로 이동 가능)
    let filteredData = members.contents;
    if (selectedLevels.length > 0) {
      filteredData = filteredData.filter((member) => selectedLevels.includes(member.level));
    }

    return filteredData.map((member) => ({
      id: member.userId,
      columns: [
        member.employeeNum,
        member.name,
        member.department,
        member.role,
        member.level.toString(),
        <ButtonPrimary
          key={member.userId}
          btnName="변경"
          styleName="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded"
          handleClick={() => openModal(member)}
        />,
      ],
    }));
  };

  useEffect(() => {
    dispatch(
      fetchMembersByPage({
        page,
        size,
        sortBy,
        sortDir,
        searchType: searchValue ? searchType : undefined,
        searchValue: searchValue || undefined,
      })
    );
  }, [dispatch, page, size, sortBy, sortDir, searchType, searchValue, selectedLevels]);

  if (loading) {
    return <div className="flex justify-center items-center h-screen">로딩 중...</div>;
  }

  return (
    <div className="p-6">
      <div className="flex items-center space-x-4">
        <div className="flex-grow">
          <InputSearch
            placeholder="회원 검색..."
            searchTypes={["이름", "사번", "부서", "권한", "레벨"]}
            onIconClick={handleSearch}
          />
        </div>

        <div className="flex items-center space-x-4 bg-white p-4 rounded-lg shadow">
          <span className="text-gray-700 font-medium">레벨 :</span>
          <div className="flex space-x-4">
            {levelOptions.map((option) => (
              <label key={option.id} className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={selectedLevels.includes(option.id)}
                  onChange={() => handleLevelChange(option.id)}
                  className="form-checkbox h-4 w-4 text-blue-500"
                />
                <span className="text-gray-700">{option.label}</span>
              </label>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow overflow-hidden">
        <TableWithAvatarAndButton
          columns={header}
          rows={getTableData()}
          iconPaths={{}}
          onIconClick={() => {}}
          width="50px"
        />
      </div>

      <div className="flex justify-center mt-6">
        <Paging curPage={page} onPageChange={setPage} totalPage={members.totalPages} />
      </div>

      {isModalOpen && (
        <Modal onClose={closeModal}>
          <div className="p-6 space-y-4 h-[50svh]">
            <h2 className="text-xl font-semibold mb-4">회원 정보 변경</h2>

            <div className="space-y-2">
              <label className="block text-gray-700">회원 레벨</label>
              <DropdownBase items={["1", "2", "3"]} buttonLabel={newLevel} onSelect={setNewLevel} width="100%" />
            </div>

            <div className="space-y-2">
              <label className="block text-gray-700">회원 권한</label>
              <DropdownBase items={roleOptions} buttonLabel={newRole} onSelect={setNewRole} width="100%" />
            </div>

            <div className="flex justify-end space-x-3 mt-6">
              <ButtonPrimary
                btnName="취소"
                styleName="bg-gray-500 hover:bg-gray-600 text-white"
                handleClick={closeModal}
              />
              <ButtonPrimary
                btnName="저장"
                styleName="bg-blue-500 hover:bg-blue-600 text-white"
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

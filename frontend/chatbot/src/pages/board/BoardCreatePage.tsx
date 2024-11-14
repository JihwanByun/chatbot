import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import SembotLayout from "@pages/SembotLayout";
import TreeCreate from "@pages/board/TreeCreate"; // TreeCreate 컴포넌트 import
import { useAppDispatch, useAppSelector } from "@app/hooks";
import BoardCreateForm, { Category, FormData } from "@pages/board/BoardCreateForm";
import { createTree } from "@app/slices/treeSlice";
import { BoardRequest } from "@apis/board/boardApi";
import { errorAlert, successAlert } from "@util/alert";
import { getCategoryListAPI } from "@apis/category/categoryApi";

const BoardCreatePage: React.FC = () => {
  // 스타일 정의
  const footStyle =
    "flex bg-transparent text-white py-2 px-4 rounded mx-1 transform hover:translate-x-1 transition-all duration-200 cursor-pointer";
  const boardButtonStyle =
    "flex bg-transparent border border-white text-white py-2 px-4 rounded mx-1 hover:bg-blue-900 transition-colors duration-100 ease-in-out";

  const dispatch = useAppDispatch();
  const navigate = useNavigate();
  // Redux store에서 상태 가져오기
  const { favorites } = useAppSelector((state) => state.favoriteBoards);

  const [formData, setFormData] = useState<FormData>({
    title: '',
    category: '',
    level: 1
  });

  const [categories, setCategories] = useState<Category[]>([]);

  const fetchCategories = async () => {
    const categories = await getCategoryListAPI();
    setCategories(categories);
    setFormData(prev => ({
      ...prev,
      category: categories[0].value
    }));
  }

  useEffect(()=>{
    fetchCategories();
  },[]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async () => {
    // API 호출 또는 다른 제출 로직
    console.log('제출된 데이터:', formData);

		try {
      const request : BoardRequest = {
        title : formData.title,
        category : formData.category,
        level : formData.level
      }
			// createTree 액션 디스패치
			const result = await dispatch(createTree(request)).unwrap();

			// 성공 처리
			console.log("Tree saved successfully:", result);
			successAlert("규정이 등록되었습니다.")
      navigate("/board");
			// 여기에 성공 메시지나 리다이렉션 로직 추가
		} catch (error) {
			console.error("Failed to save tree:", error);
			errorAlert(new Error("규정 등록에 실패하였습니다."));
			// 에러 메시지 표시 로직 추가
		}
  };

  const getChildren = () => (
    <div className="bg-white px-6">
      {/* 메인 컨텐츠 */}
      <div className="shadow-md rounded-lg text-left mb-6">
        <BoardCreateForm 
          formData={formData} 
          categories={categories} 
          onInputChange={handleInputChange}
        />
        <div className="rounded-md">
          <TreeCreate />
        </div>
      </div>
      {/** 버튼 */}
      <div className="flex justify-end">
        <button 
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          onClick={handleSubmit}
        >
          저장
        </button>
      </div>
    </div>
  );

  const sidebarComponents = [
    {
      btnName: "규정 목록",
      styleName: boardButtonStyle,
      icon: "/src/assets/icons/book-open-text-footer.svg",
      handleClick: () => navigate(`/board`),
    },
    ...favorites.map((favorite) => ({
      btnName: favorite.title,
      styleName: boardButtonStyle,
      icon: "/src/assets/icons/book-open-text-footer.svg",
      handleClick: () => navigate(`/board/${favorite.boardId}`),
    })),
  ];

  const footerComponents = [
    {
      btnName: "채팅",
      styleName: footStyle,
      icon: "/src/assets/icons/chatting-icon.svg",
    },
    {
      btnName: "로그아웃",
      styleName: footStyle,
      icon: "/src/assets/icons/logout.svg",
    },
  ];

  return (
    <SembotLayout
      title="규정 작성"
      sidebarComponents={sidebarComponents}
      footerComponents={footerComponents}
      children={getChildren()}
    />
  );
};

export default BoardCreatePage;

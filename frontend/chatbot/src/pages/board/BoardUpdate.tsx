import React, { useEffect, useState } from "react";
import TreeCreate from "@pages/board/TreeCreate"; // TreeCreate 컴포넌트 import
import { useAppDispatch } from "@app/hooks";
import BoardCreateForm, { Category, FormData } from "@pages/board/BoardCreateForm";
import { saveTreeChange, setTreeData } from "@app/slices/treeSlice";
import { BoardRequest } from "@apis/board/boardApi";
import { errorAlert, successAlert } from "@util/alert";
import { getCategoryListAPI } from "@apis/category/categoryApi";
import { BoardDetailResponse } from "@apis/board/boardDetailApi";

export interface BoardUpdatePageProps {
  board : BoardDetailResponse;
  onUpdate: () => void
}

const BoardUpdatePage: React.FC<BoardUpdatePageProps> = ({
  board,
  onUpdate
}) => {
  const dispatch = useAppDispatch();

  const [formData, setFormData] = useState<FormData>({
    title: '',
    category: '',
    level: 1
  });

  const [categories, setCategories] = useState<Category[]>([]);

  const fetchCategories = async () => {
    const categories = await getCategoryListAPI();
    setCategories(categories);
  }

  const initRequlationData = () => {
    dispatch(setTreeData(board.regulationResponseDto));
  }

  const initFormData = () => {
    setFormData({
      title: board.title,
      category: board.category,
      level: board.level
    });
  }

  useEffect(()=>{
    fetchCategories();
    initRequlationData();
    initFormData();
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
			// saveTreeChange 액션 디스패치
			const result = await dispatch(saveTreeChange({boardId:board.boardId, request})).unwrap();

			// 성공 처리
			console.log("Tree saved successfully:", result);
			successAlert("규정이 수정되었습니다.")
      onUpdate();
		} catch (error) {
			console.error("Failed to save tree:", error);
			errorAlert(new Error("규정 등록에 실패하였습니다."));
		}
  };

  return (
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
};

export default BoardUpdatePage;

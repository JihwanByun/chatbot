import React, { useEffect, useState } from "react";
import TreeCreate from "@pages/board/TreeCreate"; // TreeCreate 컴포넌트 import
import { useAppDispatch } from "@app/hooks";
import BoardCreateForm, { Category, FormData } from "@pages/board/BoardCreateForm";
import { saveTreeChange, setTreeData } from "@app/slices/treeSlice";
import { BoardRequest } from "@apis/board/boardApi";
import { errorAlert, successAlert } from "@util/alert";
import { getCategoryListAPI } from "@apis/category/categoryApi";
import { BoardDetailResponse, downloadFileAPI } from "@apis/board/boardDetailApi";
import ButtonOnlyIcon from "@components/atoms/button/ButtonOnlyIcon";
import FileUploadForm from "@components/atoms/input/InputFile";

export interface BoardUpdateProps {
  board : BoardDetailResponse
  onUpdate: () => void
  onBackClick: () => void
}

const BoardUpdate: React.FC<BoardUpdateProps> = ({
  board,
  onUpdate,
  onBackClick
}) => {
  const dispatch = useAppDispatch();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);

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
  
  const initFileData = async () => {
    if (board.hasFile && board.fileUrl) {
      try {
        const blob = await downloadFileAPI(board.fileUrl);
        const fileName = board.fileUrl.split("/").pop() || "download.pdf";
        
        // Blob을 File 객체로 변환
        const file = new File([blob], fileName, {
          type: blob.type || 'application/pdf',
          lastModified: new Date().getTime()
        });
        
        setSelectedFile(file);
      } catch (error) {
        console.error("Failed to initialize file:", error);
        errorAlert(new Error("파일 정보를 불러오는데 실패했습니다."));
      }
    }
  };

  useEffect(()=>{
    fetchCategories();
    initRequlationData();
    initFormData();
    initFileData();
  },[]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type === 'application/pdf') {
      setSelectedFile(file);
    } else {
      errorAlert(new Error("PDF 파일만 업로드 가능합니다."));
      e.target.value = '';
    }
  };

  const handleSubmit = async () => {
    // API 호출 또는 다른 제출 로직
    console.log('제출된 데이터:', formData);

		try {
      const request : BoardRequest = {
        title : formData.title,
        category : formData.category,
        level : formData.level,
        file: selectedFile
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
      {/**메인 컨텐츠 */}
      <div className="rounded-lg text-left mb-6 relative">
        <ButtonOnlyIcon
          key="move-prev-board"
          icon="/src/assets/icons/go-to-prev.svg"
          width={18}
          styleName="p-2 hover:bg-gray-100 rounded absolute right-0 top-4"
          onClick={onBackClick}
        />
        <BoardCreateForm 
          formData={formData} 
          categories={categories} 
          onInputChange={handleInputChange}
        />
        <div className="mt-4">
          {!board.hasFile ? <TreeCreate /> : 
          <FileUploadForm
            selectedFile={selectedFile}
            setSelectedFile={setSelectedFile}
            handleFileChange={handleFileChange}
          />}
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

export default BoardUpdate;

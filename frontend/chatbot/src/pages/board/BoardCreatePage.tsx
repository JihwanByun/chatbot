import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import SembotLayout from "@pages/SembotLayout";
import TreeCreate from "@pages/board/TreeCreate";
import { useAppDispatch, useAppSelector } from "@app/hooks";
import BoardCreateForm, { Category, FormData as BoardFormData } from "@pages/board/BoardCreateForm";
import { createTree, setTreeData } from "@app/slices/treeSlice";
import { BoardRequest } from "@apis/board/boardApi";
import { errorAlert, successAlert } from "@util/alert";
import { getCategoryListAPI } from "@apis/category/categoryApi";
import ButtonOnlyIcon from "@components/atoms/button/ButtonOnlyIcon";
import FileUploadForm from "@components/atoms/input/InputFile";

type InputType = 'manual' | 'file';

const BoardCreatePage: React.FC = () => {
  const footStyle =
    "flex bg-transparent text-white py-2 px-4 rounded mx-1 transform hover:translate-x-1 transition-all duration-200 cursor-pointer";
  const boardButtonStyle =
    "flex bg-transparent border border-white text-white py-2 px-4 rounded mx-1 hover:bg-blue-900 transition-colors duration-100 ease-in-out";

  const dispatch = useAppDispatch();
  const navigate = useNavigate();
  const { favorites } = useAppSelector((state) => state.favoriteBoards);

  const [inputType, setInputType] = useState<InputType>('manual');
  const [formData, setFormData] = useState<BoardFormData>({
    title: '',
    category: '',
    level: 1
  });
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [categories, setCategories] = useState<Category[]>([]);

  const fetchCategories = async () => {
    const categories = await getCategoryListAPI();
    setCategories(categories);

    if(categories.length === 0) {
      errorAlert(new Error("카테고리가 없습니다."),()=>navigate("/board", {replace : true}));    
    } else {
      setFormData(prev => ({
        ...prev,
        category: categories[0].value
      }));
    }
  }

  useEffect(()=>{
    fetchCategories();
    dispatch(setTreeData(null));
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
    try {
      const request: BoardRequest = {
        title: formData.title,
        category: formData.category,
        level: formData.level,
        file: selectedFile
      }

      const result = await dispatch(createTree(request)).unwrap();
      console.log("Tree saved successfully:", result);

      successAlert("규정이 등록되었습니다.");
      navigate("/board");
    } catch (error) {
      console.error("Failed to save:", error);
      errorAlert(new Error("규정 등록에 실패하였습니다."));
    }
  };

  const getChildren = () => (
    <div className="bg-white px-6">
      <div className="rounded-lg text-left mb-6 relative">
        <ButtonOnlyIcon
          key="move-prev-board"
          icon="/src/assets/icons/go-to-prev.svg"
          width={18}
          styleName="p-2 hover:bg-gray-100 rounded absolute right-0 top-4"
          onClick={()=>navigate(-1)}
        />
        <BoardCreateForm 
          formData={formData} 
          categories={categories} 
          onInputChange={handleInputChange}
        />
        {/** 입력 방식 토글 버튼 */}
        <div className="mt-4">
          <div className="flex">
            <button
              className={`py-2 px-4 font-medium focus:outline-none border rounded-b-none border-b-white ${
                inputType === 'manual'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-gray-200 text-gray-500 hover:text-gray-700 hover:border-gray-400 hover:border-b-white'
              }`}
              onClick={() => setInputType('manual')}
            >
              직접 입력
            </button>
            <button
              className={`py-2 px-4 font-medium focus:outline-none border rounded-b-none border-b-white ${
                inputType === 'file'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-gray-200 text-gray-500 hover:text-gray-700 hover:border-gray-400 hover:border-b-white'
              }`}
              onClick={() => setInputType('file')}
            >
              파일 업로드
            </button>
          </div>
        </div>
        {/** 규정 입력 or 파일 입력 */}
        <div className="border border-t-0 rounded-t-none rounded-md px-6 py-6">
          {inputType === 'manual' ? <TreeCreate /> : 
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
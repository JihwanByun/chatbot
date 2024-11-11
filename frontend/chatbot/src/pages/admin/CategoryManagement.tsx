import React, { useEffect, useState } from "react";
import CardWithButton from "@components/atoms/card/CardWithButton";
import ButtonWithIcon from "@components/atoms/button/ButtonWithIcon";
import { addCategory, removeCategory, fetchCategories } from "@apis/chat/manageCategory";
import { stringify } from "flatted"; // 순환 참조를 안전하게 처리
import { categoryDTO } from "./categoryDTO";

const CategoryManagement = () => {
  const [contents, setContents] = useState<Array<{ categoryId: number; name: string }>>([]); // 카테고리 리스트
  const [newCategory, setNewCategory] = useState(""); // 새 카테고리 입력
  const [isAdding, setIsAdding] = useState(false); // 카테고리 추가 입력창 열기

  // 페이지 로드 시 전체 카테고리 조회
  useEffect(() => {
    const loadCategories = async () => {
      try {
        const response = await fetchCategories(); // 전체 조회 API 호출
        console.log("Fetched categories:", response.data); // 응답 데이터 확인
        // 응답 형식 반영: categories 배열을 상태에 저장
        setContents(Array.isArray(response.data.categories) ? response.data.categories : []);
      } catch (error) {
        console.error("카테고리 조회 실패:", error);
      }
    };
    loadCategories();
  }, []);

  const handleAddCategory = async (newCategory: string) => {
    if (!newCategory.trim()) {
      console.error("카테고리 이름이 비어있습니다.");
      return;
    }
  
    try {
      console.log(newCategory);
  
      // 카테고리 추가
      const response = await addCategory(newCategory);
  
      // 새 카테고리 추가 후 상태 업데이트
      // response.data에서 필요한 정보를 바로 사용
      setContents((prevContents) => [
        ...prevContents,
        { categoryId: response.data.categoryId, name: response.data.name },
      ]);
  
      setNewCategory(""); // 입력 필드 초기화
      setIsAdding(false); // 추가 창 닫기
    } catch (error) {
      console.error("카테고리 추가 실패:", error);
    }
  };
  

  // 카테고리 삭제 처리
  const handleRemoveCategory = async (categoryId: number) => {
    try {
      await removeCategory(categoryId); // 서버에서 삭제 요청
      // 삭제된 카테고리 상태에서 제거
      setContents((prevContents) => prevContents.filter((content) => content.categoryId !== categoryId));
    } catch (error) {
      console.error("카테고리 삭제 실패:", error);
    }
  };

  return (
    <div className="flex flex-col items-center space-y-4">
      {contents && contents.map((content) => (
        <div className="w-[40%]" key={content.categoryId}>
          <CardWithButton
            content={content.name}
            handleClick={() => handleRemoveCategory(content.categoryId)} // 클릭 시 categoryId 전달
          />
        </div>
      ))}
      
      {/* 카테고리 추가 입력창 */}
      {isAdding ? (
        <div className="w-[40%] flex flex-col items-center space-y-2">
            <input
              type="text"
              value={newCategory}
              onChange={(e) => setNewCategory(e.target.value)}
              placeholder="새 카테고리 이름"
              className="p-3 border rounded w-full text-lg"  // 크기 조정을 위한 스타일
            />
          <div className="flex space-x-2 mt-2">
            <ButtonWithIcon
              btnName="추가"
              styleName="py-2 px-4 rounded-lg bg-gray-200 whitespace-nowrap"
              icon="src/assets/icons/plus.svg"  // 아이콘 경로 수정
              isFooter={true}
              handleClick={() => handleAddCategory(newCategory)} // 추가 버튼 클릭 시 추가 처리
            />
            <ButtonWithIcon
              btnName="취소"
              styleName="py-2 px-4 rounded-lg bg-gray-200 whitespace-nowrap"
              icon="src/assets/icons/x-circle.svg"  // 아이콘 경로 수정
              isFooter={true}
              handleClick={() => {
                setIsAdding(false);  // 추가 창 닫기
                setNewCategory("");  // 입력 필드 초기화
              }}
            />
          </div>
        </div>
      ) : (
        <ButtonWithIcon
          btnName="카테고리 추가"
          styleName="w-[40%] py-2 rounded-lg bg-blue-500 text-white"
          icon="/src/assets/icons/plus.svg"  // 아이콘 경로 수정
          isFooter={true}
          handleClick={() => setIsAdding(true)} // 카테고리 추가 버튼 클릭 시 입력창 열기
        />
      )}
    </div>
  );
};

export default CategoryManagement;

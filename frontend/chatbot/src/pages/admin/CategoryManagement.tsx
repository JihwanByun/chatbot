import React, { useState } from "react";
import CardWithButton from "@components/atoms/card/CardWithButton";
import ButtonWithIcon from "@components/atoms/button/ButtonWithIcon";
import { addCategory, removeCategory, updateCategory } from "@apis/chat/manageCategory";  // Importing the API functions

const initialContents = ["뉴진스♪", "아이브♨", "에스파♥", "프로미스나인★"];

const CategoryManagement = () => {
  const [contents, setContents] = useState(initialContents);  // Initial category list
  const [newCategory, setNewCategory] = useState("");  // New category input
  const [isAdding, setIsAdding] = useState(false);  // State to toggle adding new category
  const [editingIndex, setEditingIndex] = useState<number | null>(null); // Track which category is being edited

  const handleClick = (index: number) => {
    setContents((prevContents) => prevContents.filter((_, i) => i !== index));  // Delete category
  };

  const handleAddCategory = async () => {
    try {
      const response = await addCategory(newCategory);  // Call API to add category
      setContents((prevContents) => [...prevContents, response.data.name]);  // Update state with the new category
      setNewCategory("");  // Clear input field
      setIsAdding(false);  // Close input field
    } catch (error) {
      console.error("카테고리 추가 실패:", error);  // Log errors
    }
  };

  const handleRemoveCategory = async (index: number) => {
    const categoryToRemove = contents[index];
    try {
      await removeCategory(categoryToRemove);  // Call API to remove category
      setContents((prevContents) => prevContents.filter((_, i) => i !== index));  // Update state to reflect removal
    } catch (error) {
      console.error("카테고리 삭제 실패:", error);  // Log errors
    }
  };

  const handleEditCategory = async (index: number) => {
    const newCategoryName = prompt("새 카테고리 이름을 입력하세요", contents[index]);
    if (newCategoryName && newCategoryName !== contents[index]) {
      try {
        await updateCategory(contents[index], newCategoryName);  // Call API to update category
        setContents((prevContents) => {
          const updatedContents = [...prevContents];
          updatedContents[index] = newCategoryName;  // Update the category name in state
          return updatedContents;
        });
      } catch (error) {
        console.error("카테고리 수정 실패:", error);  // Log errors
      }
    }
  };

  return (
    <div className="flex flex-col items-center space-y-4">
      {contents.map((content, index) => (
        <div className="w-[40%]" key={index}>
          <CardWithButton
            content={content}
            handleClick={() => handleClick(index)}  // Trigger removal on click
          />
          <div className="flex space-x-2 mt-2">
            {/* Edit and Delete Buttons */}
            <ButtonWithIcon
              btnName="수정"
              styleName="py-1 px-2 rounded-lg bg-yellow-300"
              icon="/src/assets/icons/edit.svg"
              isFooter={false}
              handleClick={() => handleEditCategory(index)}  // Trigger edit functionality
            />
            <ButtonWithIcon
              btnName="삭제"
              styleName="py-1 px-2 rounded-lg bg-red-500 text-white"
              icon="/src/assets/icons/delete.svg"
              isFooter={false}
              handleClick={() => handleRemoveCategory(index)}  // Trigger removal functionality
            />
          </div>
        </div>
      ))}
      {/* Add Category Section */}
      {isAdding ? (
        <div className="w-[40%] flex flex-col items-center space-y-2">
          <input
            type="text"
            value={newCategory}
            onChange={(e) => setNewCategory(e.target.value)}
            placeholder="새 카테고리 이름"
            className="p-2 border rounded"
          />
          <ButtonWithIcon
            btnName="추가"
            styleName="py-2 rounded-lg bg-blue-500 text-white"
            icon="/src/assets/icons/plus.svg"
            isFooter={true}
            handleClick={handleAddCategory}  // Trigger category addition
          />
          <ButtonWithIcon
            btnName="취소"
            styleName="py-2 rounded-lg bg-gray-300"
            icon="/src/assets/icons/cancel.svg"
            isFooter={true}
            handleClick={() => setIsAdding(false)}  // Cancel addition
          />
        </div>
      ) : (
        <ButtonWithIcon
          btnName="카테고리 추가"
          styleName="w-[40%] py-2 rounded-lg bg-gray-200"
          icon="/src/assets/icons/plus.svg"
          isFooter={true}
          handleClick={() => setIsAdding(true)}  // Open input field for adding category
        />
      )}
    </div>
  );
};

export default CategoryManagement;

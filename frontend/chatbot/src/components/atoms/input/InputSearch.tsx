import React, { useState } from "react";
import ButtonWithIcon from "../button/ButtonOnlyIcon";
import Dropdown from "../dropdown/Dropdown";

interface InputSearchProps {
  /** Input 필드에 표시될 메시지 */
  placeholder?: string;
  /** Input란의 너비 */
  width?: string;
  /** Input란의 글자 색 */
  textColor?: string;
  /** Input란의 배경 색 */
  backgroundColor?: string;
  /** Input란의 Border 색 */
  borderColor?: string;
  /** icon 경로 */
  iconPath?: string;
  /** icon click event 함수 */
  onIconClick?: (searchType: string, searchValue: string) => void;
  /** 검색 타입 리스트 */
  searchTypes?: string[];
  /** 추가될 속성값 */
  className?: string;
}

const InputSearch: React.FC<InputSearchProps> = ({
  /** 기본 값 할당 */
  placeholder = "Search",
  width = "full",
  textColor = "#000000",
  backgroundColor = "#FFFFFF",
  borderColor = "#000000",
  iconPath = "/src/assets/icons/search.svg",
  onIconClick,
  searchTypes = ["제목", "작성자"],
  className,
}) => {
  const [inputValue, setInputValue] = useState<string>("");
  const [selectedType, setSelectedType] = useState<string>(searchTypes[0]);

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(event.target.value);
  };

  const handleTypeSelect = (type: string) => {
    setSelectedType(type);
  };

  const handleSearch = () => {
    if (onIconClick) {
      onIconClick(selectedType, inputValue.trim());
    }
  };

  // Enter 키 처리
  const handleKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      handleSearch();
    }
  };

  return (
    <div
      className="flex items-center space-x-2 relative text-gray-600 w-full border border-gray-300 rounded-md"
      style={{ width }}
    >
      <Dropdown
        items={searchTypes}
        buttonLabel={selectedType}
        width="10rem"
        className="border-none"
        onSelect={handleTypeSelect}
      />
      <input
        type="text"
        value={inputValue}
        onChange={handleInputChange}
        onKeyPress={handleKeyPress}
        placeholder={placeholder}
        className="rounded-md h-12 w-full px-5 pr-10 text-sm focus:outline-none focus:ring-0"
        style={{ color: textColor, backgroundColor, borderColor }}
      />

      <div
        className={`absolute right-0 top-1/2 transform -translate-y-1/2 pr-3 ${className}`}
      >
        <ButtonWithIcon icon={iconPath} onClick={handleSearch} />
      </div>
    </div>
  );
};

export default InputSearch;

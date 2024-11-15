import React, { useEffect, useState } from "react";

interface InputWithIconProps {
  /** Input란에 띄워질 메세지 */
  placeholder?: string;
  bgColor?: string;
  borderColor?: string;
  textColor?: string;
  /** icon 경로 */
  iconPath: string;
  /** icon click event 함수 */
  onIconClick: (message: string) => void;
  isLoading? : boolean;
  className?: string;
  initInputValue?: string;
}

const InputWithIcon: React.FC<InputWithIconProps> = ({
  /** 기본 값 할당 */
  placeholder = "",
  bgColor = "#EAF1FB",
  textColor = "#434343",
  iconPath,
  onIconClick,
  isLoading = false,
  className,
  initInputValue = ""
}) => {
  const [inputValue, setInputValue] = useState<string>(initInputValue);
  const loadingIconPath ="/src/assets/icons/loading.svg";

  useEffect(() => {
    setInputValue(initInputValue);
  }, [initInputValue]);

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(event.target.value);
  };

  const onClick = () => {
    if(!isLoading){
      setInputValue("");
      onIconClick(inputValue);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      onClick();
    }
  };

  return (
    <div className="relative w-full">
      <input
        type="text"
        value={inputValue}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        style={{ 
          backgroundColor: bgColor || 'white',
          color: textColor || 'rgb(17 24 39)'
        }}
        className={`
          h-12 w-full px-5 pr-12
          text-base font-medium
          rounded-lg
          border-2 border-semesBlue/30
          transition-all duration-100
          placeholder:text-gray-400
          focus:outline-0 focus:border-semesBlue/80
          hover:border-semesBlue/70
          shadow-[0_2px_8px_-3px_rgba(0,0,0,0.1)]
          ${className}
        `}
      />
      
      {/* 버튼 영역 */}
      <div className="absolute right-1.5 top-1/2 -translate-y-1/2">
        <button
          onClick={onClick}
          className="p-2 rounded-md
            hover:bg-semesBlue/10 
            active:bg-semesBlue/20 
            transition-colors"
          aria-label="Search"
        >
          <img 
            src={isLoading ? loadingIconPath : iconPath} 
            alt="icon"
            width={18}
            className="opacity-80 hover:opacity-100 transition-opacity" 
          />
        </button>
      </div>
    </div>
  );
};

export default InputWithIcon;

import React, { useCallback, useEffect, useRef, useState } from "react";

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

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const lineHeight = 20;
  const remainHeight = 28;
  const maxLines = 15;

  useEffect(() => {
    setInputValue(initInputValue);
  }, [initInputValue]);

  const handleInputChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(event.target.value);
  };

  const onClick = () => {
    if(!isLoading){
      setInputValue("");
      onIconClick(inputValue);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter") {
      if(event.shiftKey || event.ctrlKey) return;
      event.preventDefault();
      onClick();
    }
  };

  // textarea 높이 자동 조절
  const adjustHeight = useCallback(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const numLines = textarea.value.split('\n').length;

    // 15줄 이하일 때는 컨텐츠 높이만큼, 초과하면 maxHeight로 제한
    const newHeight = remainHeight + lineHeight * Math.min(numLines, maxLines);
    
    // 높이 적용
    textarea.style.height = `${newHeight}px`;

    // 스크롤은 15줄 초과시에만 표시
    textarea.style.overflowY = numLines > maxLines ? 'auto' : 'hidden';
  }, []);

  useEffect(() => {
    adjustHeight();
  }, [inputValue, adjustHeight]);

  return (
    <div className="relative w-full">
      <textarea
        ref={textareaRef}
        value={inputValue}
        onChange={(e) => {
          handleInputChange(e);
          adjustHeight();
        }}
        onKeyDown={(e) => {
          handleKeyDown(e);
        }}
        placeholder={placeholder}
        style={{ 
          backgroundColor: bgColor || 'white',
          color: textColor || 'rgb(17 24 39)',
          resize: 'none',
          overflowY: 'hidden',
        }}
        className={`
          w-full px-5 pr-12 py-3
          text-base font-medium
          rounded-lg
          border-2 border-semesBlue/30
          transition-colors duration-100
          placeholder:text-gray-400
          focus:outline-0 focus:border-semesBlue/80
          hover:border-semesBlue/70
          shadow-[0_2px_8px_-3px_rgba(0,0,0,0.1)]
          leading-5
          ${className}
        `}
      />
    
      {/* 버튼 영역 */}
      <div 
        className={`
          absolute bottom-3 right-3 
          transition-all duration-200
        `}
      >
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

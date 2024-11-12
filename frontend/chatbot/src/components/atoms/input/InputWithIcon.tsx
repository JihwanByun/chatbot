import React, { useEffect, useState } from "react";
import ButtonWithIcon from "../button/ButtonOnlyIcon";

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
  borderColor = "#004F9F",
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
    <div className="relative text-gray-800 w-full">
      <input
        type="text"
        value={inputValue}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        style={{ backgroundColor: bgColor, borderColor, color: textColor }}
        className={`rounded-md h-12 w-full px-5 pr-10 text-base border focus:outline-none focus:ring-0 ${className}`}
      />
      <div className="absolute right-0 top-1/2 transform -translate-y-1/2 pr-3">
        <ButtonWithIcon icon={isLoading ? loadingIconPath : iconPath} onClick={onClick} width={18} />
      </div>
    </div>
  );
};

export default InputWithIcon;

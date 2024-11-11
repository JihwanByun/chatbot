interface ButtonProps {
	icon: string;
	styleName: string;
	onClick: () => void;
	width?: number;  // width를 선택적인 속성으로 추가
  }
  
  const ButtonWithIcon = ({ icon, styleName, width = 24, onClick }: ButtonProps) => {  // 기본값을 24로 설정
	return (
	  <div data-ripple-light="true" className={styleName} onClick={onClick}>
		<img src={icon} width={width}></img>
	  </div>
	);
  };
  
  export default ButtonWithIcon;
  
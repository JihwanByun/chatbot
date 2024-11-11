export type ButtonProps = {
	icon: string;
	styleName?: string;
	width?: number;
	// TODO
	onClick?: () => void;
	disabled: boolean;
};


const ButtonOnlyIcon = ({ icon, styleName, width = 24, onClick }: ButtonProps) => {  
	return (
	  <div data-ripple-light="true" className={styleName} onClick={onClick}>
		<img src={icon} width={width} />  {/* width를 설정하여 아이콘 크기를 변경 */}
	  </div>
	);
  };

 
  
export default ButtonOnlyIcon;

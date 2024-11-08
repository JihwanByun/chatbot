export type ButtonProps = {
	icon: string;
	styleName?: string;
	width?: string;
	// TODO
	onClick?: () => void;
	disabled: boolean;
};

const ButtonWithIcon = ({ icon, styleName, width, onClick }: ButtonProps) => {
	return (
		<div data-ripple-light="true" className={styleName} onClick={onClick}>
			<img src={icon} width={width}></img>
		</div>
	);
};

export default ButtonWithIcon;

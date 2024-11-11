import React from "react";

export interface ButtonProps {
	btnName: string;
	styleName: string;
	icon: string;
	handleClick?: () => void;
	isActive?: boolean;
	isFooter?: boolean;
	iconStyleName? : string;
	handleIconClick? : () => void;
}

const ButtonWithIcon: React.FC<ButtonProps> = ({
	btnName,
	styleName,
	icon,
	handleClick,
	isActive,
	isFooter,
	iconStyleName,
	handleIconClick = () => {}
}) => {
	const activeStyle = isActive
		? "bg-white text-semesBlue"
		: "bg-gray-400 text-semesBlue";

	const footerStyle = isFooter ? "bg-semesBlue text-white" : "";
	const combinedStyle = `space-x-2 ${styleName} ${activeStyle}`;
	const combinedFooterStyle = `space-x-2 ${styleName} ${footerStyle}`;

	return (
		<div
			className={footerStyle == "" ? combinedStyle : combinedFooterStyle}
			data-ripple-light="true"
			onClick={handleClick}
		>
			<div className="flex">
				<img width="15rem" src={icon} onClick={(e)=>{e.stopPropagation(); handleIconClick();}} className={`mr-2 ${iconStyleName}`}></img>
				<div>{btnName}</div>
			</div>
		</div>
	);
};

export default ButtonWithIcon;

import React from "react";
import { useNavigate } from "react-router-dom";

export interface ButtonProps {
	btnName: string;
	styleName: string;
	icon: string;
	handleClick?: () => void;
	isActive?: boolean;
	isFooter?: boolean;
}

const ButtonWithIcon: React.FC<ButtonProps> = ({
	btnName,
	styleName,
	icon,
	handleClick,
	isActive,
	isFooter,
}) => {
	const activeStyle = isActive
		? "bg-white text-semesBlue"
		: "bg-gray-400 text-semesBlue";

	const navigate = useNavigate();
	const handleClickDefault = () => {
		switch (btnName) {
			case "규정 확인하기":
				navigate("/board");
				break;
			case "로그아웃":
				localStorage.removeItem("Authorization");
				localStorage.removeItem("Role");
				navigate("/");
				// TODO : 로그아웃 api 호출해야함
				break;
			case "채팅":
				navigate("/chat");
				break;

			default:
		}
	};

	const footerStyle = isFooter ? "bg-semesBlue text-white" : "";
	const combinedStyle = `space-x-2 ${styleName} ${activeStyle} cursor-pointer`;
	const combinedFooterStyle = `space-x-2 ${styleName} ${footerStyle} cursor-pointer`;

	return (
		<div
			className={footerStyle == "" ? combinedStyle : combinedFooterStyle}
			data-ripple-light="true"
			onClick={handleClick ? handleClick : handleClickDefault}
		>
			<div className="flex">
				<div className="w-1"></div>
				<img width="15rem" src={icon}></img>
				<div className="w-2"></div>
				<div>{btnName}</div>
			</div>
		</div>
	);
};

export default ButtonWithIcon;

import React, { useState } from "react";
import { useAppSelector } from "@app/hooks";
import { useNavigate } from "react-router-dom";
import { useAppDispatch } from "@app/hooks";
import { logoutUser } from "@app/slices/userSlice";

interface HeaderProps {
	/** 해당 페이지 정보 */
	title?: string;
	/** 사용자 이미지 */
	userImage?: string;
}

const Header: React.FC<HeaderProps> = ({
	/** 기본값 */
	userImage = "/src/assets/icons/user-profile-ex.svg",
	title = "전체",
}) => {
	const dispatch = useAppDispatch();
	const navigate = useNavigate();
	const [isProfileDropdownOpen, setIsProfileDropdownOpen] = useState(false);

	// Redux store에서 사용자 정보 가져오기
	const { name, employeeNum, department } = useAppSelector((state) => state.users);

	const toggleProfileDropdown = () => {
		setIsProfileDropdownOpen(!isProfileDropdownOpen);
	};

	const handleLogout = async () => {
		try {
			await dispatch(logoutUser()).unwrap();
			navigate("/login");
		} catch (error) {
			console.error("로그아웃 실패:", error);
		}
	};

	return (
		<header>
			<div className="relative mt-5 w-full flex items-center mx-auto py-4">
				<div className="absolute sm:left-5 text-2xl font-bold left-16">{title}</div>

				<div className="absolute flex items-center right-0">
					{/* 회원 정보 */}
					<div className="flex flex-col items-end gap-1">
						<div className="text-xs text-gray-600">{employeeNum}</div>
						<div className="text-xs font-medium">{name}님</div>
					</div>

					{/* 회원 이미지 */}
					<button
						type="button"
						className="bg-transparent relative flex rounded-full focus:outline-none"
						onClick={toggleProfileDropdown}
					>
						<img className="h-8 w-8 rounded-full" src={userImage} alt="프로필 이미지" />
					</button>
				</div>

				{/* Dropdown component */}
				{isProfileDropdownOpen && (
					<div className="absolute right-0 top-8 mr-8 mt-2 px-6 origin-top-right rounded-md py-1 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none items-center text-center z-50 bg-white">
						<div className="block px-4 py-2 text-sm text-gray-700 border-b border-gray-100">
							<p className="font-medium">{name}님</p>
							<p className="text-gray-500">{department}</p>
						</div>
						<button
							onClick={handleLogout}
							className="block w-full px-4 py-2 text-sm text-left text-gray-700 hover:bg-gray-100"
						>
							로그아웃
						</button>
					</div>
				)}
			</div>
		</header>
	);
};

export default Header;
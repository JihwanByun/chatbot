import React, { useState, useRef, useEffect } from "react";
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

// 랜덤 파스텔 컬러 생성 함수
const getRandomPastelColor = (str: string) => {
	let hash = 0;
	for (let i = 0; i < str.length; i++) {
		hash = str.charCodeAt(i) + ((hash << 5) - hash);
	}

	// 파스텔톤을 위해 채도와 명도를 조정
	const h = hash % 360;
	return `hsl(${h}, 70%, 85%)`;
};

// Avatar 컴포넌트
const Avatar: React.FC<{
	name: string;
	image?: string;
	className?: string;
}> = ({ name, image, className = "" }) => {
	if (image) {
		return (
			<img
				src={image}
				alt={`${name}의 프로필`}
				className={`h-10 w-10 rounded-full object-cover ${className}`}
			/>
		);
	}

	const initial = name.charAt(0).toUpperCase();
	const backgroundColor = getRandomPastelColor(name);

	return (
		<div
			className={`h-10 w-10 rounded-full flex items-center justify-center ${className}`}
			style={{
				backgroundColor,
				color: '#444',
				fontSize: '1.2rem',
				fontWeight: '600'
			}}
		>
			{initial}
		</div>
	);
};

const Header: React.FC<HeaderProps> = ({
	/** 기본값 */
	userImage,
	title = "전체",
}) => {
	const dispatch = useAppDispatch();
	const navigate = useNavigate();
	const [isProfileDropdownOpen, setIsProfileDropdownOpen] = useState(false);
	const dropdownRef = useRef<HTMLDivElement>(null);

	// Redux store에서 사용자 정보 가져오기
	const { name, employeeNum, department } = useAppSelector((state) => state.users);

	// 드롭다운 외부 클릭 감지
	useEffect(() => {
		const handleClickOutside = (event: MouseEvent) => {
			if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
				setIsProfileDropdownOpen(false);
			}
		};

		document.addEventListener('mousedown', handleClickOutside);
		return () => document.removeEventListener('mousedown', handleClickOutside);
	}, []);

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
		<header className="bg-white">
			<div className="max-w-8xl mx-auto">
				<div className="relative flex justify-between items-center h-10">
					{/* 타이틀 영역 */}
					<div className="absolute left-16 sm:left-5 flex-1 flex items-center">
						<h1 className="text-2xl font-bold text-gray-800 tracking-wide">
							{title}
						</h1>
					</div>

					{/* 사용자 정보 영역 */}
					<div className="absolute right-0 flex items-center" ref={dropdownRef}>
						{/* 사용자 텍스트 정보 */}
						<div className="hidden sm:flex flex-col items-end">
							<div className="text-sm font-medium text-gray-700">{employeeNum}</div>
							<div className="text-sm font-semibold text-gray-900">{name} 님</div>
						</div>

						{/* 프로필 버튼 */}
						<div className="relative">
							<button
								type="button"
								className="flex text-sm transition-transform duration-200 hover:scale-105"
								onClick={toggleProfileDropdown}
							>
								<Avatar
									name={name}
									image={userImage}
									className="transition-transform duration-200 hover:scale-105"
								/>
							</button>

							{/* 드롭다운 메뉴 */}
							{isProfileDropdownOpen && (
								<div className="absolute right-0 mt-2 w-48 rounded-lg shadow-lg py-1 bg-white ring-1 ring-black ring-opacity-5 z-50 transform transition-all duration-200 ease-out">
									{/* 프로필 헤더 */}
									<div className="px-4 py-3 border-b border-gray-100">
										<div className="flex items-start py-2">
											<Avatar name={name} image={userImage} className="h-8 w-8 border-none" />
											<div className="flex flex-col items-start justify-start pl-2">
												<p className="text-sm font-semibold text-gray-900 truncate">
													{name}
												</p>
												<p className="text-xs text-gray-500 truncate">
													{department}
												</p>
											</div>
										</div>
									</div>

									{/* 로그아웃 버튼 */}
									<button
										onClick={handleLogout}
										className="w-full text-left block px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors duration-150"
									>
										로그아웃
									</button>
								</div>
							)}
						</div>
					</div>
				</div>
			</div>
		</header>
	);
};

export default Header;
import React, { useState } from "react";

interface HeaderProps {
	/** 해당 페이지 정보 */
	title?: string;
	/** 사용자 이미지 */
	userImage?: string;
	/** 사용자 이름 */
	userName?: string;
	/** 사용자 사번 */
	userNumber?: string;
}

const Header: React.FC<HeaderProps> = ({
	/** 기본값 */
	userImage = "/src/assets/icons/user-profile-ex.svg",
	title = "전체",
	userName = "SemBot",
	userNumber = "12345678",
}) => {
	const [isProfileDropdownOpen, setIsProfileDropdownOpen] = useState(false);

	const toggleProfileDropdown = () => {
		setIsProfileDropdownOpen(!isProfileDropdownOpen);
	};

  return (
    <header>
      <div className="relative mt-5 w-full flex items-center mx-auto py-4">
        <div className="absolute sm:left-5 text-2xl font-bold left-16">{title}</div>

        <div className="absolute flex items-center right-0">
          {/* 회원 정보 */}
          <div className="items-end end-0 justify-end">
            <div className="text-xs">{userNumber}</div>
            <div className="text-xs">{userName}님</div>
          </div>

					{/* 회원 이미지 */}
					<button
						type="button"
						className="bg-transparent relative flex rounded-full focus:outline-none"
						onClick={toggleProfileDropdown}
					>
						<img className="h-8 w-8 rounded-full" src={userImage} alt="" />
					</button>
				</div>

        {/* Dropdown component */}
        {isProfileDropdownOpen && (
          <div className="absolute right-0 top-8 mr-8 mt-2 px-2 origin-top-right rounded-md py-1 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none items-center text-center z-50 bg-white">
            <div className="block px-2 py-2 text-sm">
              {userName}님 환영합니다!
            </div>
            <a href="#" className="block px-4 py-2 text-sm">
              Sign out
            </a>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;

import { useEffect, useState } from "react";
import { LoginDTO } from "./LoginDTO";
import logo from "@/assets/images/logo-login.png";
import topLeftLogo from "@/assets/images/head-logo-group.png";
import ButtonPrimary from "@components/atoms/button/ButtonPrimary";
import { useAppDispatch, useAppSelector } from "@app/hooks"; // Redux hooks 사용
import { loginUser } from "@app/slices/userSlice"; // loginUser thunk import
import { useNavigate } from "react-router-dom";
import { UserRole } from "@util/userConfig";

const LoginForm: React.FC = () => {
	const navigate = useNavigate();
	const [email, setEmail] = useState("");
	const [password, setPassword] = useState("");
	const [rememberMe, setRememberMe] = useState(false);
	const dispatch = useAppDispatch();
	const { role, loading, error } = useAppSelector((state) => state.users); // 상태 조회

	const handleSubmit = (e: React.FormEvent) => {
		e.preventDefault();
		const formData: LoginDTO = { email, password };
		dispatch(loginUser(formData)); // loginUser thunk를 dispatch하여 로그인 요청
	};

  useEffect(() => {
    // role이 존재하고 빈 문자열이 아닐 때만 navigate
    if (role) {
      if (role === UserRole.ADMIN) {
        navigate("/adminPage", { replace: true });  // replace 옵션 추가
      } else {
        navigate("/chat", { replace: true });
      }
    }
  }, [role, navigate]);

	const handleSignUpClick = () => {
		navigate("/register");
	};

	return (
		<div className="flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
			{/* Top-left logo */}
			<img
				src={topLeftLogo}
				alt="Top Left Logo"
				className="absolute top-4 left-4 w-16 h-auto"
			/>

			<div className="max-w-md w-full p-4 bg-white shadow-md rounded">
				<div className="text-center mb-6">
					<img
						src={logo}
						alt="Logo"
						className="mx-auto mb-2 w-full max-h-32 h-auto object-contain"
					/>
				</div>

				{/* 에러 메시지 출력 */}
				{error && (
					<div className="text-red-500 text-sm mb-4 text-center">{error}</div>
				)}

				<form onSubmit={handleSubmit} className="flex flex-col space-y-4">
					<div className="mb-4">
						<label
							htmlFor="email"
							className="block text-sm font-medium text-gray-700 text-left"
						>
							이메일
						</label>
						<input
							type="email"
							placeholder="Email"
							value={email}
							onChange={(e) => setEmail(e.target.value)}
							className="border p-2 w-full"
							required
						/>
					</div>

					<div className="mb-4">
						<label
							htmlFor="password"
							className="block text-sm font-medium text-gray-700 text-left"
						>
							비밀번호
						</label>
						<input
							type="password"
							placeholder="Password"
							value={password}
							onChange={(e) => setPassword(e.target.value)}
							className="border p-2 w-full"
							required
						/>
					</div>

					{/* Remember Me checkbox */}
					<div className="flex items-center mb-4">
						<input
							type="checkbox"
							checked={rememberMe}
							onChange={(e) => setRememberMe(e.target.checked)}
							className="mr-2"
						/>
						<label className="text-sm text-gray-700">
							로그인 정보 기억하기
						</label>
					</div>

					{/* Login and Sign Up buttons */}
					<div className="flex w-full">
						{/* Left button - 로그인 */}
						<div className="flex-1">
							<ButtonPrimary
								btnName={loading ? "로딩 중..." : "로그인"}
								handleClick={() => handleSubmit}
								styleName="bg-[#004F9F] text-white py-2 px-4 rounded w-full"
								isDisabled={!email || !password || loading}
							/>
						</div>
						{/* Right button - 회원가입하기 */}
						<div className="flex-1">
							<ButtonPrimary
								btnName="회원가입하기"
								handleClick={handleSignUpClick}
								styleName="bg-white text-[#004F9F] py-2 px-4 rounded w-full border"
								isDisabled={false}
							/>
						</div>
					</div>
				</form>
			</div>
		</div>
	);
};

export default LoginForm;

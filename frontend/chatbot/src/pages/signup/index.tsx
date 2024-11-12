import React, { useState } from "react";
import { emailDTO, SignUpDTO } from "./SignUpDTO";
import logo from "@/assets/images/head-register.png";
import topLeftLogo from "@/assets/images/head-logo-group.png"; // Import your top-left logo
import ButtonPrimary from "@components/atoms/button/ButtonPrimary";
import { signUp } from "@apis/chat/signup";
import { checkEmailDuplicate } from "@apis/chat/checkEmailDuplicate";
import { useNavigate } from "react-router-dom";

export interface SignUpFormProps {
	onSubmit: (formData: SignUpDTO) => void;
}

const SignUpForm: React.FC = () => {
	const [email, setEmail] = useState("");
	const [name, setName] = useState("");
	const [employeeNum, setEmployeeNum] = useState("");
	const [department, setDepartment] = useState("");
	const [password, setPassword] = useState("");
	const [passwordVerify, setPasswordVerify] = useState("");
	const [termsAccepted, setTermsAccepted] = useState(false);

	//이메일 중복 확인 여부를 저장하는 상태
	const [isEmailChecked, setIsEmailChecked] = useState(false);

	const navigate = useNavigate();

	const handleSubmit = async (e: React.FormEvent) => {
		e.preventDefault();

		if (!termsAccepted) {
			alert("이용 약관에 동의해 주세요.");
			return;
		}

		const formData: SignUpDTO = {
			email,
			name,
			employeeNum,
			department,
			password,
			passwordVerify,
		};

		try {
			const response = await signUp(formData);
			console.log("회원가입 성공", response);
			alert("회원가입에 성공했습니다!, 로그인 페이지로 이동합니다.");
			navigate("/chat");
		} catch (error) {
			console.error("회원가입 실패", error);
			alert("회원가입에 실패했습니다. 다시 시도해주세요.");
		}
	};

	const handleEmailCheck = async (e: React.FormEvent) => {
		e.preventDefault();

		const formData: emailDTO = { email };

		try {
			const response = await checkEmailDuplicate(formData);
			// email 변수를 올바르게 연결
			if (response != null) {
				const isDuplicate = await response;

				if (isDuplicate) {
					alert("이메일이 이미 사용 중입니다."); // 중복된 경우
					setIsEmailChecked(false); // 중복된 경우 체크를 취소
				} else {
					alert("사용 가능한 이메일입니다."); // 중복되지 않은 경우
					setIsEmailChecked(true); // 중복 확인 완료
				}
			} else {
				alert("이메일 중복 확인 중 오류가 발생했습니다."); // 서버 오류
			}
		} catch (error) {
			console.error("이메일 중복 확인 중 오류 발생:", error);
			alert("이메일 중복 확인 중 오류가 발생했습니다. 다시 시도해 주세요.");
		}
	};

	return (
		<div className="flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
			{/* Top-left logo */}
			<img
				src={topLeftLogo}
				alt="Top Left Logo"
				className="absolute top-4 left-4 w-16 h-auto"
			/>
			<div className="max-w-md mx-auto p-4 bg-white shadow-md rounded relative z-10">
				<div className="text-center mb-6">
					<img
						src={logo}
						alt="Logo"
						className="mx-auto mb-2 w-full max-h-32 h-auto object-contain"
					/>
				</div>

				<form onSubmit={handleSubmit} className="flex flex-col space-y-4">
					<div className="mb-4">
						<label
							htmlFor="email"
							className="block text-sm font-medium text-gray-700 text-left"
						>
							이메일
						</label>
						<div className="flex items-center">
							<input
								type="email"
								placeholder="Email"
								value={email}
								onChange={(e) => {
									setEmail(e.target.value);
									setIsEmailChecked(false);
								}}
								className="border p-2 flex-grow"
								required
							/>
							<ButtonPrimary
								btnName="중복 확인"
								styleName="ml-2 bg-[#004F9F] text-white text-xs p-2 rounded hover:bg-blue-700" // Reduced font size
								handleClick={handleEmailCheck}
							/>
						</div>
					</div>

					<div className="mb-4">
						<label
							htmlFor="name"
							className="block text-sm font-medium text-gray-700 text-left"
						>
							이름
						</label>
						<input
							type="text"
							placeholder="Name"
							value={name}
							onChange={(e) => setName(e.target.value)}
							className="border p-2 w-full"
							required
						/>
					</div>

					<div className="mb-4">
						<label
							htmlFor="name"
							className="block text-sm font-medium text-gray-700 text-left"
						>
							사번
						</label>
						<input
							type="text"
							placeholder="EmployeeNumber"
							value={employeeNum}
							onChange={(e) => setEmployeeNum(e.target.value)}
							className="border p-2 w-full"
							required
						/>
					</div>
					<div className="mb-4">
						<label
							htmlFor="department"
							className="block text-sm font-medium text-gray-700 text-left"
						>
							부서
						</label>
						<select
							id="department"
							value={department}
							onChange={(e) => setDepartment(e.target.value)}
							required
							className="mt-1 p-2 border border-gray-300 rounded w-full"
						>
							<option value="">선택</option>
							<option value="인사">HR</option>
							<option value="IT">Engineering</option>
							<option value="마켓팅">Marketing</option>
							<option value="영업">Sales</option>
						</select>
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
					<div className="mb-4">
						<label
							htmlFor="passwordVerify"
							className="block text-sm font-medium text-gray-700 text-left"
						>
							비밀번호 확인
						</label>
						<input
							type="password"
							placeholder="Password"
							value={passwordVerify}
							onChange={(e) => setPasswordVerify(e.target.value)}
							className="border p-2 w-full"
							required
						/>
					</div>
					<div className="mb-4 flex items-center">
						<input
							type="checkbox"
							id="terms"
							checked={termsAccepted}
							onChange={() => setTermsAccepted(!termsAccepted)}
							required
						/>
						<label htmlFor="terms" className="ml-2 text-sm text-gray-600">
							이용 약관의 동의합니다.
						</label>
					</div>
					<button
						type="submit"
						className="bg-[#004F9F] text-white p-2 rounded hover:bg-blue-700"
						disabled={!isEmailChecked || !termsAccepted}
					>
						회원가입
					</button>
				</form>
			</div>
		</div>
	);
};

export default SignUpForm;

import React, { useState } from "react";
import { emailDTO, SignUpDTO } from "./SignUpDTO";
import logo from "@assets/images/head-register.png";
import topLeftLogo from "@assets/images/head-logo-group.png";
import ButtonPrimary from "@components/atoms/button/ButtonPrimary";
import { signUp } from "@apis/chat/signup";
import { checkEmailDuplicate } from "@apis/chat/checkEmailDuplicate";
import { useNavigate } from "react-router-dom";
import Alert, { AlertProps } from "@components/atoms/alert/Alert";

const SignUpForm: React.FC = () => {
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [employeeNum, setEmployeeNum] = useState("");
  const [department, setDepartment] = useState("");
  const [password, setPassword] = useState("");
  const [passwordVerify, setPasswordVerify] = useState("");
  const [termsAccepted, setTermsAccepted] = useState(false);
  const [isEmailChecked, setIsEmailChecked] = useState(false);
  const [alertProps, setAlertProps] = useState<AlertProps>({
    showAlert: false,
    icon: "success",
    title: "",
    text: "",
  });

  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!isEmailChecked) {
      setAlertProps({
        showAlert: true,
        icon: "warning",
        title: "이메일 확인 필요",
        text: "이메일 중복 확인을 해주세요.",
      });
      return;
    }

    if (!termsAccepted) {
      setAlertProps({
        showAlert: true,
        icon: "warning",
        title: "약관 동의 필요",
        text: "이용 약관에 동의해 주세요.",
      });
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
      setAlertProps({
        showAlert: true,
        icon: "success",
        title: "회원가입 성공!",
        text: "로그인 페이지로 이동합니다.",
        onClose: () => navigate("/login"),
      });
    } catch (error) {
      console.error("회원가입 실패", error);
      setAlertProps({
        showAlert: true,
        icon: "error",
        title: "회원가입 실패",
        text: "회원가입에 실패했습니다. 다시 시도해주세요.",
      });
    }
  };

  // 이메일 중복 확인을 위한 별도의 핸들러 함수
  const handleEmailCheckClick = async (e: React.MouseEvent) => {
    e.preventDefault(); // 폼 제출 방지

    if (!email) {
      setAlertProps({
        showAlert: true,
        icon: "warning",
        title: "이메일 입력 필요",
        text: "이메일을 입력해 주세요.",
      });
      return;
    }

    const formData: emailDTO = { email };

    try {
      setAlertProps((prevProps) => ({
        ...prevProps,
        showAlert: false,
      }));

      const response = await checkEmailDuplicate(formData);
      if (response != null) {
        const isDuplicate = await response;

        if (isDuplicate) {
          setAlertProps({
            showAlert: true,
            icon: "error",
            title: "중복된 이메일",
            text: "이메일이 이미 사용 중입니다.",
          });
          setIsEmailChecked(false);
        } else {
          setAlertProps({
            showAlert: true,
            icon: "success",
            title: "사용 가능",
            text: "사용 가능한 이메일입니다.",
          });
          setIsEmailChecked(true);
        }
      }
    } catch (error) {
      setAlertProps({
        showAlert: true,
        icon: "error",
        title: "오류 발생",
        text: "이메일 중복 확인 중 오류가 발생했습니다. 다시 시도해 주세요.",
      });
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <Alert {...alertProps} />

      <div className="absolute top-8 left-8 transition-transform hover:scale-105">
        <img src={topLeftLogo} alt="Top Left Logo" className="w-36 h-auto" />
      </div>

      <div className="w-full max-w-md transform transition-all duration-300 hover:scale-[1.01] my-8">
        <div className="bg-white rounded-2xl shadow-xl p-8 space-y-6">
          <div className="text-center transition-all duration-300 hover:transform hover:scale-105">
            <img src={logo} alt="Logo" className="mx-auto h-24 object-contain" />
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="relative space-y-2">
              <label className="text-sm font-medium text-gray-700">이메일</label>
              <div className="flex gap-2">
                <div className="relative flex-grow">
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => {
                      setEmail(e.target.value);
                      setIsEmailChecked(false);
                    }}
                    required
                    className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500 transition-colors"
                    placeholder="이메일을 입력해주세요"
                  />
                </div>
                <button
                  type="button" // form submit 방지
                  onClick={handleEmailCheckClick}
                  disabled={isEmailChecked}
                  className={`px-4 py-3 rounded-lg text-sm font-medium transition-colors duration-200 whitespace-nowrap
                    ${
                      isEmailChecked
                        ? "bg-gray-400 cursor-not-allowed text-white"
                        : "bg-blue-600 hover:bg-blue-700 text-white"
                    }`}
                >
                  중복 확인
                </button>
              </div>
            </div>

            <div className="relative">
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500 transition-colors peer"
                placeholder=" "
              />
              <label className="absolute left-4 top-3 text-gray-500 transition-all duration-200 -translate-y-7 bg-white px-1 text-sm peer-placeholder-shown:translate-y-0 peer-placeholder-shown:text-base peer-focus:-translate-y-7 peer-focus:text-sm peer-focus:text-blue-500">
                이름
              </label>
            </div>

            <div className="relative">
              <input
                type="text"
                value={employeeNum}
                onChange={(e) => setEmployeeNum(e.target.value)}
                required
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500 transition-colors peer"
                placeholder=" "
              />
              <label className="absolute left-4 top-3 text-gray-500 transition-all duration-200 -translate-y-7 bg-white px-1 text-sm peer-placeholder-shown:translate-y-0 peer-placeholder-shown:text-base peer-focus:-translate-y-7 peer-focus:text-sm peer-focus:text-blue-500">
                사번
              </label>
            </div>

            <div className="relative">
              <select
                value={department}
                onChange={(e) => setDepartment(e.target.value)}
                required
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500 transition-colors appearance-none bg-white"
              >
                <option value="">부서를 선택해주세요</option>
                <option value="인사">HR</option>
                <option value="IT">Engineering</option>
                <option value="마켓팅">Marketing</option>
                <option value="영업">Sales</option>
              </select>
            </div>

            <div className="relative">
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500 transition-colors peer"
                placeholder=" "
              />
              <label className="absolute left-4 top-3 text-gray-500 transition-all duration-200 -translate-y-7 bg-white px-1 text-sm peer-placeholder-shown:translate-y-0 peer-placeholder-shown:text-base peer-focus:-translate-y-7 peer-focus:text-sm peer-focus:text-blue-500">
                비밀번호
              </label>
            </div>

            <div className="relative">
              <input
                type="password"
                value={passwordVerify}
                onChange={(e) => setPasswordVerify(e.target.value)}
                required
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500 transition-colors peer"
                placeholder=" "
              />
              <label className="absolute left-4 top-3 text-gray-500 transition-all duration-200 -translate-y-7 bg-white px-1 text-sm peer-placeholder-shown:translate-y-0 peer-placeholder-shown:text-base peer-focus:-translate-y-7 peer-focus:text-sm peer-focus:text-blue-500">
                비밀번호 확인
              </label>
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                id="terms"
                checked={termsAccepted}
                onChange={() => setTermsAccepted(!termsAccepted)}
                required
                className="w-4 h-4 text-blue-600 border-2 border-gray-300 rounded focus:ring-blue-500"
              />
              <label htmlFor="terms" className="ml-2 text-sm text-gray-600 hover:text-gray-800 transition-colors">
                이용 약관에 동의합니다
              </label>
            </div>

            <button
              type="submit"
              className={`w-full py-3 px-4 rounded-lg text-sm font-medium transition-colors duration-200
                ${
                  !isEmailChecked || !termsAccepted
                    ? "bg-gray-400 cursor-not-allowed"
                    : "bg-blue-600 hover:bg-blue-700 text-white"
                }`}
              disabled={!isEmailChecked || !termsAccepted}
            >
              회원가입
            </button>

            <div className="text-center pt-4">
              <p className="text-sm text-gray-600">
                이미 계정이 있으신가요?{" "}
                <button
                  type="button"
                  onClick={() => navigate("/login")}
                  className="text-blue-600 hover:text-blue-700 font-medium transition-colors"
                >
                  로그인하기
                </button>
              </p>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default SignUpForm;

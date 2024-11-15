import { useEffect, useState } from "react";
import { LoginDTO } from "./LoginDTO";
import logo from "@assets/images/logo-login.png";
import topLeftLogo from "@assets/images/head-logo-group.png";
import ButtonPrimary from "@components/atoms/button/ButtonPrimary";
import { useAppDispatch, useAppSelector } from "@app/hooks";
import { loginUser } from "@app/slices/userSlice";
import { useNavigate } from "react-router-dom";
import { UserRole } from "@util/userConfig";
import { isAuthenticated } from "@util/auth";
import Alert, { AlertProps } from "@components/atoms/alert/Alert";

const LoginForm: React.FC = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [rememberMe, setRememberMe] = useState(false);
  const dispatch = useAppDispatch();
  const { role, loading } = useAppSelector((state) => state.users);
  const [alertProps, setAlertProps] = useState<AlertProps>({
    showAlert: false,
    icon: "error",
    title: "",
    text: "",
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const formData: LoginDTO = { email, password };

    try {
      // 이전 Alert을 닫음
      setAlertProps((prev) => ({
        ...prev,
        showAlert: false,
      }));

      const result = await dispatch(loginUser(formData)).unwrap();

      // 로그인 성공 시 Alert 표시
      setAlertProps({
        showAlert: true,
        icon: "success",
        title: "로그인 성공",
        text: "환영합니다!",
        onClose: () => {
          if (role === UserRole.ADMIN) {
            navigate("/adminPage", { replace: true });
          } else {
            navigate("/chat", { replace: true });
          }
        },
      });
    } catch (error: any) {
      let errorMessage = "로그인에 실패했습니다.";
      let errorTitle = "로그인 실패";

      if (error?.response?.status === 401) {
        errorTitle = "인증 실패";
        errorMessage = "아이디 또는 비밀번호가 올바르지 않습니다.";
      } else if (error?.response?.status === 404) {
        errorTitle = "계정 없음";
        errorMessage = "등록되지 않은 이메일입니다.";
      } else if (error?.response?.status === 403) {
        errorTitle = "접근 제한";
        errorMessage = "계정이 잠겼거나 비활성화되었습니다. 관리자에게 문의하세요.";
      }

      setAlertProps({
        showAlert: true,
        icon: "error",
        title: errorTitle,
        text: errorMessage,
      });
    }
  };

  useEffect(() => {
    if (isAuthenticated() && role) {
      if (role === UserRole.ADMIN) {
        navigate("/adminPage", { replace: true });
      } else {
        navigate("/chat", { replace: true });
      }
    }
  }, [role, navigate]);

  const handleSignUpClick = () => {
    navigate("/register");
  };

  // Alert이 닫힐 때 호출되는 함수
  const handleAlertClose = () => {
    setAlertProps((prev) => ({
      ...prev,
      showAlert: false,
    }));
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      {/* Alert 컴포넌트 */}
      <Alert {...alertProps} onClose={handleAlertClose} />

      {/* 좌측 상단 logo */}
      <div className="absolute top-8 left-8 transition-transform hover:scale-105">
        <img src={topLeftLogo} alt="Top Left Logo" className="w-36 h-auto" />
      </div>

      {/* 메인 화면 */}
      <div className="w-full max-w-md transform transition-all duration-300 hover:scale-[1.01]">
        <div className="bg-white rounded-2xl shadow-xl p-8 space-y-8">
          <div className="text-center transition-all duration-300 hover:transform hover:scale-105">
            <img src={logo} alt="Logo" className="mx-auto h-24 object-contain" />
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Email 입력란 */}
            <div className="relative">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500 transition-colors peer"
                placeholder=" "
              />
              <label className="absolute left-4 top-3 text-gray-500 transition-all duration-200 -translate-y-7 bg-white px-1 text-sm peer-placeholder-shown:translate-y-0 peer-placeholder-shown:text-base peer-focus:-translate-y-7 peer-focus:text-sm peer-focus:text-blue-500">
                이메일
              </label>
            </div>

            {/* Password 입력란 */}
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

            {/* Remember Me 체크박스 */}
            <div className="flex items-center">
              <input
                type="checkbox"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
                className="w-4 h-4 text-blue-500 border-2 border-gray-300 rounded focus:ring-blue-500"
              />
              <label className="ml-2 text-sm text-gray-600 hover:text-gray-800 transition-colors">
                로그인 정보 기억하기
              </label>
            </div>

            <div className="grid grid-cols-2 gap-4 pt-4">
              <ButtonPrimary
                btnName={loading ? "로딩 중..." : "로그인"}
                handleClick={() => handleSubmit}
                styleName="bg-blue-600 hover:bg-blue-700 text-white py-3 px-4 rounded-lg transition-colors duration-200 w-full text-sm font-medium"
                isDisabled={!email || !password || loading}
              />
              <ButtonPrimary
                btnName="회원가입하기"
                handleClick={handleSignUpClick}
                styleName="bg-white text-blue-600 border-2 border-blue-600 hover:bg-blue-50 py-3 px-4 rounded-lg transition-colors duration-200 w-full text-sm font-medium"
                isDisabled={false}
              />
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default LoginForm;

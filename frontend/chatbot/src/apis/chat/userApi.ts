import defaultAxios from "@apis/common";
import { LoginDTO } from "@pages/login/LoginDTO";

export const login = async (loginDTO: LoginDTO) => {
	// API 요청 및 결과 반환
	const result = await defaultAxios.post("/api/users/login", loginDTO)
	localStorage.setItem("Authorization", result.data.token);
	return result;
};

export const logout = async () => {
	const result = await defaultAxios.post("/api/users/logout")
	localStorage.removeItem("Authorization");
	return result;
};
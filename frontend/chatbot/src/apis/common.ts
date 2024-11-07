import { BACKEND_URL, AI_URL } from "@/configs/config";
import axios, { AxiosInstance } from "axios";
// import { IncomingMessage, ServerResponse } from 'http';

// 로그인이 필요없는 axios
export const defaultAxios: AxiosInstance = axios.create({
	baseURL: `${BACKEND_URL}`,
	withCredentials: true, // 쿠키 전송 허용
});

export const defaultAIAxios: AxiosInstance = axios.create({
	baseURL: `${AI_URL}`,
	// withCredentials: true, // 쿠키 전송 허용
});

// 토큰을 설정할 수 있는 함수 추가
export const setAuthToken = (token: string | null) => {
	if (token) {
		console.log(token);
		defaultAxios.defaults.headers.common["Authorization"] = `${token}`;
	} else {
		console.log("LOGOUT!!!!");
		delete defaultAxios.defaults.headers.common["Authorization"];
	}
};

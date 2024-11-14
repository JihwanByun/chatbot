import { BACKEND_URL, AI_URL } from "@/configs/config";
import { logoutUser } from "@app/slices/userSlice";
import axios, { AxiosInstance } from "axios";
// import { IncomingMessage, ServerResponse } from 'http';

// 로그인이 필요없는 axios
const defaultAxios: AxiosInstance = axios.create({

	baseURL: `${BACKEND_URL}`,
	withCredentials: true, // 쿠키 전송 허용
});

defaultAxios.interceptors.request.use(function (config) {
	// Authorization 헤더를 추가
	config.headers['Authorization'] = localStorage.getItem("Authorization");
	
	// 수정된 config 반환
	return config;
  }, function (error) {
	// 요청 에러 처리 (선택적)
	return Promise.reject(error);
  });


export const defaultAIAxios: AxiosInstance = axios.create({
	baseURL: `${AI_URL}`,
	// withCredentials: true, // 쿠키 전송 허용
});
// 응답 인터셉터 추가
defaultAxios.interceptors.response.use(
	(response) => {
		return response;
	},
	(error) => {
		// 토큰 에러 체크
		if (error.response && error.response.status === 401 || error.response.status === 422) {
			logoutUser();
		}
		return Promise.reject(error);
	}
);	

export default defaultAxios;

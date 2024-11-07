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

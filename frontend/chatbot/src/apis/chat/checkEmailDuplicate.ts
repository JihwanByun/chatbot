import defaultAxios from "@apis/common";
import { emailDTO } from "@pages/signup/SignUpDTO";



export const checkEmailDuplicate = async (formData: emailDTO) => {
    const response = await defaultAxios.get(`/api/users?email=${formData.email}`);
    return response.data; // 중복 여부만 반환
};

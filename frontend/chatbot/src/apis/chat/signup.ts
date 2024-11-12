import defaultAxios from "@apis/common"
import { SignUpDTO } from "@pages/signup/SignUpDTO"

export const signUp = async (formData: SignUpDTO) => {
    try{

        const response = defaultAxios.post("/api/users/register", formData)
       
        return response;
    
    } catch (error) {
        console.error('회원가입 요청 오류', error);
        throw error;
    }
} 

 //      {
        //     email: formData.email,
        //     password: formData.password,
        //     name: formData.name,
        //     department:formData.department,
        //     passwordVerify: formData.passwordVerify
        // });
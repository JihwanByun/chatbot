import  defaultAxios  from "@apis/common"

export const addCategory = async (name: string) => {
    try {
        // 문자열을 그대로 보내는 방법
        const response = await defaultAxios.post(
            "/api/admins/categories",
            name, // 순수 문자열
            {
                
              headers: {
            //     'Content-Type': 'text/plain' // 단순 텍스트 형식으로 지정
                'Content-Type': 'text/plain;charset=UTF-8'
              }
            }
          );
          
        return response;
    } catch (error) {
        console.error('카테고리 추가 요청 오류', error);
        throw error;
    }
}


export const removeCategory = async (categoryId: number) => {
    const response = await defaultAxios.delete(`api/admins/${categoryId}`)

    return response;
}
export const fetchCategories = async () => {
    try {
        const response = await defaultAxios.get("/api/categories"); // 전체 카테고리 조회 요청
        
        console.log(response.data)
        return response;
    } catch (error) {
        console.error("카테고리 조회 요청 오류", error);
        throw error;
    }
};
// export const updateCategory = async (categoryId: number, newCategoryName: String) => {
//     const response = defaultAxios.put(`api/admins/${categoryId}`, newCategoryName)

//     return response;
// }


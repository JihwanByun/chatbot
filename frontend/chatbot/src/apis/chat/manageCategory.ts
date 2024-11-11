import { defaultAxios } from "@apis/common"


export const addCategory = async (newCategory: String) => {
    try{

        const response = defaultAxios.post("/api/admins/categories", newCategory)

        return response;
    } catch (error) {
        console.error('카테고리 추가 요청 오류', error);
        throw error;
    }
}

export const removeCategory = async (categoryId: number) => {
    const response = defaultAxios.delete(`api/admins/${categoryId}`)

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


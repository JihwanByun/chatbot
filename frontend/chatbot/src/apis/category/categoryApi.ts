import defaultAxios from "@apis/common";
import { ChatCategory } from "@components/chat/ChatCategories";

type CategoryResponse = {
  categoryId: number;
  name: string
}

type CategoryListResponse = {
  categories: CategoryResponse[]
}

// export interface ChatCategory {
//   name: string;
//   prompt: string;
// }

const convertToChatCategory = (res : CategoryResponse) : ChatCategory => {
  return {
    name: res.name,
    prompt: res.name
  }
}

const convertToChatCategoryList = (res : CategoryListResponse) : ChatCategory[] => {
  return res.categories.map(convertToChatCategory);
}

export const getCategoryListAPI = async () : Promise<ChatCategory[]> => {
  return defaultAxios
        .get<CategoryListResponse>("/api/categories")
        .then(res => convertToChatCategoryList(res.data))
}
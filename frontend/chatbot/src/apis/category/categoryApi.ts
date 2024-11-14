import defaultAxios from "@apis/common";
import { ChatCategory } from "@components/chat/ChatCategories";
import { Category } from "@pages/board/BoardCreateForm";

export type CategoryResponse = {
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

export const getChatCategoryListAPI = async () : Promise<ChatCategory[]> => {
  return defaultAxios
        .get<CategoryListResponse>("/api/categories")
        .then(res => convertToChatCategoryList(res.data))
}

const convertToCategory = (res : CategoryResponse) : Category => {
  return {
    value: res.name,
    label: res.name,
  }
}

const convertTotCategoryList = (res : CategoryListResponse) : Category[] => {
  return res.categories.map(convertToCategory);
}

export const getCategoryListAPI = async () : Promise<Category[]> => {
  return defaultAxios
        .get<CategoryListResponse>("/api/categories")
        .then(res => convertTotCategoryList(res.data))
}
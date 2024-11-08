import { defaultAIAxios, defaultAxios } from "@apis/common";
import { BaseMessage } from "@components/chat/ChatMessage";
import { Doc, DocMetadata, QnA } from "@components/chat/ChatView";
import { AI_URL } from "@configs/config";

//===== AI API =====//
type SearchDocResponse = {
  docs : DocResponse[]
}

type DocResponse = {
  metadata : DocMetadata;
  page_content : string;
}

const convertToDoc = (res : DocResponse) : Doc => {
  return {
    metadata : res.metadata,
    content : res.page_content
  }
}

const convertToDocArray = (responses: DocResponse[]): Doc[] => {
  return responses.map(convertToDoc);
};

// 문서 검색
export const searchDocsAPI = async (question : string): Promise<Doc[]> => {
  return defaultAIAxios
    .post<SearchDocResponse>('search', { question })
    .then((res) => {
      return convertToDocArray(res.data.docs);
    })
    .catch((error) => {
      console.error("Error in searchDocsAPI:", error);
      return [];
    });
};

type Memory = {
  question : string;
  doc : Doc | null;
  answer : string | null;
}

// type GenerateResponse = {
//   answer : string;
// }

const convertToMemory = (qna : QnA) : Memory => {
  return {
    question: qna.question.content,
    doc: qna.docs && qna.docs.length > 0 ? qna.docs[0] : null,
    answer: qna.answer.content || null
  }
}

const convertToMemoryArray = (qnas : QnA[]) : Memory[] => {
  return qnas.map(convertToMemory);
}

export const generateAPI = async (qnas: QnA[], question: string): Promise<Response> => {
  return await fetch(`${AI_URL}/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ memory: convertToMemoryArray(qnas), question })
  });
};

//===== Backend API =====//

export type ChatroomResponse = {
  chatRoomId: number,
  title: string,
  createdAt: string
}

type ChatroomResponsePage = {
  content: ChatroomResponse[]
}

type ChatroomListResponse = {
  contents: ChatroomResponsePage,
  hasNext: boolean
}

export type ChatroomList = {
  contents: ChatroomResponse[],
  hasNext: boolean
}

const convertToChatroomList = (chatroomListResponse : ChatroomListResponse) : ChatroomList => {
  return {
    contents: chatroomListResponse.contents.content,
    hasNext: chatroomListResponse.hasNext
  };
}

export const getChatroomListAPI = async (page : number = 0): Promise<ChatroomList | null> => {
  return defaultAxios
    .get<ChatroomListResponse>(`/api/chatrooms`,{
      headers: {
        Authorization: localStorage.getItem("Authorization"),
      },
      params: {
        page
      },
    })    
    .then((res) => {
      return convertToChatroomList(res.data);
    })
    .catch((error) => {
      console.error("Error in getChatroomListAPI:", error);
      return null;
    });
};


export const createChatroomAPI = async (message : string): Promise<ChatroomResponse | null> => {
  return defaultAxios
    .post<ChatroomResponse>('/api/chatrooms', { content : message }, {
      headers: {
        Authorization: localStorage.getItem("Authorization"),
      },
    })
    .then((res) => {
      return res.data;
    })
    .catch((error) => {
      console.error("Error in createChatroomAPI:", error);
      return null;
    });
};


export type DocMetadataBE = {
  fileName: string,
  totalPages: number | null,
  page: number,
  category: string | null,
  id: number | null
}

export type DocBE = {
  metaData: DocMetadataBE
  pageContent: string
}

export type MemoryBE = {
  question: string,
  docs: DocBE[],
  answer: string
}

export type ChatroomDetailResponse = {
  chatRoomId: number,
  createdAt: string,
  chatList: ChatResponse[]
}

export type ChatResponse = {
  chatId: string,
  memory: MemoryBE,
  isPositive: boolean
}

export type ChatroomDetail = {
  chatroomId: number,
  createdAt: string,
  qnas: QnA[]
}

const convertToDocMetadata = (metaData : DocMetadataBE) : DocMetadata => {
  return {
    source : metaData.fileName,
    page : metaData.page
  }
}

const convertDocBEToDoc = (res : DocBE) : Doc => {
  return {
    metadata : convertToDocMetadata(res.metaData),
    content : res.pageContent
  }
}

const convertDocBEArrToDocArray = (responses: DocBE[]): Doc[] => {
  return responses.map(convertDocBEToDoc);
};

const convertToQnA = (chat : ChatResponse) : QnA => {
  return {
    chatId: chat.chatId,
    question: new BaseMessage("question", chat.memory.question),
    answer: new BaseMessage("answer", chat.memory.answer),
    docs: convertDocBEArrToDocArray(chat.memory.docs),
    isPositive: chat.isPositive,
    isAnswered: true
  };
}

const convertToQnAArray = (chats : ChatResponse[]) : QnA[] => {
  return chats.map(convertToQnA);
}

const convertToChatroomDetail = (res : ChatroomDetailResponse) : ChatroomDetail => {
  return {
    chatroomId: res.chatRoomId,
    createdAt: res.createdAt,
    qnas: convertToQnAArray(res.chatList).reverse()
  }
}

export const getChatroomDetailAPI = async (chatroomId : number): Promise<ChatroomDetail | null> => {
  return defaultAxios
    .get<ChatroomDetailResponse>(`/api/chatrooms/${chatroomId}`,{
      headers: {
        Authorization: localStorage.getItem("Authorization"),
      }
    })    
    .then((res) => {
      return convertToChatroomDetail(res.data);
    })
    .catch((error) => {
      console.error("Error in getChatroomDetailAPI:", error);
      return null;
    });
};

const convertToDocMetadataBE = (metadata : DocMetadata) : DocMetadataBE => {
  return {
    fileName : metadata.source,
    page : metadata.page,
    totalPages: null,
    category: null,
    id: null
  }
}

const convertDocToDocBE = (res : Doc) : DocBE => {
  return {
    metaData : convertToDocMetadataBE(res.metadata),
    pageContent : res.content
  }
}
const convertDocArrToDocBEArr = (responses: Doc[]): DocBE[] => {
  return responses.map(convertDocToDocBE);
};

const convertToMemoryBE = (qna : QnA) : MemoryBE => {
  return {
    question: qna.question.content,
    docs: convertDocArrToDocBEArr(qna.docs),
    answer: qna.answer.content
  }
}

// const convertToMemoryBEArray = (qnas : QnA[]) : MemoryBE[] => {
//   return qnas.map(convertToMemoryBE);
// }

export const createChatAPI = async (chatroomId : number, qna : QnA, ): Promise<ChatResponse | null> => {
  return defaultAxios
    .post<ChatResponse>('/api/chats', 
      {
        chatRoomId: chatroomId,
        memory: convertToMemoryBE(qna),
        isPositive: qna.isPositive
      },
      {
        headers: {
          Authorization: localStorage.getItem("Authorization"),
        },
      }
    )
    .then((res) => {
      return res.data;
    })
    .catch((error) => {
      console.error("Error in createChatAPI:", error);
      return null;
    });
};

type ChatFeedbackResponse = {
  chat: ChatResponse,
  negativeReason: string
}

export const createChatFeedbackAPI = async (chatId : string, isPositive : boolean, negativeReason?: string): Promise<ChatFeedbackResponse | null> => {
  return defaultAxios
    .post<ChatFeedbackResponse>(`/api/chats/${chatId}/feedback`, 
      {
        isPositive,
        negativeReason
      },
      {
        headers: {
          Authorization: localStorage.getItem("Authorization"),
        },
      }
    )
    .then((res) => {
      return res.data;
    })
    .catch((error) => {
      console.error("Error in createChatFeedbackAPI:", error);
      return null;
    });
};

// {
//   "chatRoomId": 1,
//   "createdAt": "2024-11-07T16:57:33.795579",
//   "chatList": []
// }
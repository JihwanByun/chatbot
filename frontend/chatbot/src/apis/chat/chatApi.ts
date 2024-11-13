import defaultAxios, { defaultAIAxios } from "@apis/common";
import { BaseMessage } from "@components/chat/ChatMessage";
import { Doc, DocMetadata, QnA } from "@components/chat/ChatView";
import { AI_URL } from "@configs/config";

//===== AI API =====//
type SearchDocResponse = {
  docs : DocResponse[] | boolean
}

type DocMetadataAI = {
  file_name: string;
  source?: string;
  page?: number;
  level: number;
}

type DocResponse = {
  metadata : DocMetadataAI;
  page_content : string;
}

const convertDocMetadataAIToDocMetadata = (metaData : DocMetadataAI) : DocMetadata => {
  return {
    filename: metaData.file_name,
    source: metaData.source,
    page: metaData.page,
    level: metaData.level
  }
}

const convertToDoc = (res : DocResponse) : Doc => {
  return {
    metadata : convertDocMetadataAIToDocMetadata(res.metadata),
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
      const docs = res.data.docs;

      if(typeof docs === 'boolean') throw new Error("검색하려는 자료에 대한 접근 권한이 없습니다.");

      return docs && typeof docs !== 'boolean' ? convertToDocArray(docs) : docs;
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

export const generateAPI = async (qnas: QnA[], question: string, timeout = 30000): Promise<Response> => {
  
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  try{
    const response = await fetch(`${AI_URL}/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ memory: convertToMemoryArray(qnas), question }),
      signal: controller.signal
    });
    clearTimeout(id);
    
  return response;
  } catch (error){
    if(error === "AbortError"){
      throw new Error("요청에 실패했습니다.");
    }
    throw error;
  }

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

export const getChatroomListAPI = async (page : number = 0): Promise<ChatroomList> => {
  return defaultAxios
    .get<ChatroomListResponse>(`/api/chatrooms`,{
      params: {
        page
      },
    })    
    .then((res) => {
      return convertToChatroomList(res.data);
    })
    .catch((error) => {
      console.error("Error in getChatroomListAPI:", error);
      throw new Error("채팅방 목록을 가져오는데 실패했습니다.");
    });
};


export const createChatroomAPI = async (message : string): Promise<ChatroomResponse> => {
  return defaultAxios
    .post<ChatroomResponse>('/api/chatrooms', { content : message })
    .then((res) => {
      return res.data;
    })
    .catch((error) => {
      console.log(error);
      throw new Error(`채팅방 생성에 실패했습니다.`);
    });
};


export type DocMetadataBE = {
  fileName: string,
  totalPages?: number,
  page?: number,
  category?: string,
  id?: number
  level: number
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
    filename : metaData.fileName,
    page : metaData.page,
    level: metaData.level
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

export const getChatroomDetailAPI = async (chatroomId : number): Promise<ChatroomDetail> => {
  return defaultAxios
    .get<ChatroomDetailResponse>(`/api/chatrooms/${chatroomId}`)    
    .then((res) => {
      return convertToChatroomDetail(res.data);
    })
    .catch((error)=>{
      console.log(error);
      throw new Error("채팅방 정보를 가져오는데 실패했습니다.");
    });
};

const convertToDocMetadataBE = (metadata : DocMetadata) : DocMetadataBE => {
  return {
    fileName : metadata.filename,
    page : metadata.page,
    level: metadata.level
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

export const createChatAPI = async (chatroomId : number, qna : QnA, ): Promise<ChatResponse> => {
  return defaultAxios
    .post<ChatResponse>('/api/chats', 
      {
        chatRoomId: chatroomId,
        memory: convertToMemoryBE(qna),
        isPositive: qna.isPositive
      }
    )
    .then((res) => {
      return res.data;
    })
    .catch((error)=>{
      console.log(error);
      throw new Error("채팅 저장에 실패했습니다.");
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
      }
    )
    .then((res) => {
      return res.data;
    });
};

export const deleteChatroomAPI = async(chatroomId : number) => {
  return defaultAxios
  .delete(`/api/chatrooms/${chatroomId}`)
  .then((res) => {
    return res.data;
  });
}

// {
//   "chatRoomId": 1,
//   "createdAt": "2024-11-07T16:57:33.795579",
//   "chatList": []
// }
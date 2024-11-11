import defaultAxios from "@apis/common";

type FeedbackReasonResponse = {
  name: string;
}

const convertToStringList = (responses: FeedbackReasonResponse[]) => {
  return responses.map(res=>res.name);
}

export const getFeedbackReasonList = async() : Promise<string[]> => {
  return defaultAxios
        .get<FeedbackReasonResponse[]>("/api/feedbacks")
        .then(res=>convertToStringList(res.data));
}
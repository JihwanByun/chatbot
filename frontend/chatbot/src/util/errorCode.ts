export const Error = {
  INVALID_TOKEN : {
    status : 422,
    code : "INVALID_TOKEN"
  },
  EXPIRED_TOKEN : {
    status : 401,
    code : "EXPIRED_TOKEN"
  },
  UNSUPPORTED_TOKEN : {
    status : 400,
    code : "UNSUPPORTED_TOKEN"
  },
  NO_REFRESH_TOKEN : {
    status : 404,
    code : "NO_REFRESH_TOKEN"
  },
}
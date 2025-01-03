# Build Stage
FROM node:20-alpine as build

WORKDIR /app

# Install dependencies first (better caching)
COPY package*.json ./
RUN npm ci

# Copy source code
COPY . .

# 빌드 시 환경변수 받기
ARG VITE_BACKEND_URL
ARG VITE_AI_URL

# 환경변수 설정
ENV VITE_BACKEND_URL=$VITE_BACKEND_URL
ENV VITE_AI_URL=$VITE_AI_URL

# Build app
# ENV CI=false
RUN npm run build

# Production Stage
FROM nginx:alpine

# Copy built assets from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# assets 폴더 복사
COPY --from=build /app/src/assets /usr/share/nginx/html/src/assets

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
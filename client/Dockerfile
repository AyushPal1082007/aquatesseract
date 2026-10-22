FROM node:20.15.1-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci || npm install --package-lock=false --no-audit --no-fund
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev"]

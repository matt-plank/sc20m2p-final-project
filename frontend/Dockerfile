FROM node:16-alpine

WORKDIR /app

# Copy fiels to container
COPY package*.json ./
RUN npm install
COPY . .

# Build static files from source
RUN npm run build

EXPOSE 80

# Serve static files with nginx
CMD ["nginx", "-g", "daemon off;"]

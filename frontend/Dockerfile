FROM node:16-alpine

WORKDIR /app

# Copy fiels to container
COPY package*.json ./
RUN npm install
COPY . .

EXPOSE 3000

# Serve static files with nginx
CMD ["npm", "start"]

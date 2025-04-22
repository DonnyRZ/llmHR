# Use an official Nginx image (alpine is small)
FROM nginx:stable-alpine

# Copy your frontend code (HTML, CSS, JS, images) to the default Nginx static content directory
COPY ./ /usr/share/nginx/html

# Nginx listens on port 80 by default, expose it inside the container network
EXPOSE 80

# Nginx image has a default command to start the server, so no CMD needed here
# This version is not compatible with Apple M chips
# To build for Apple M chips or ARM architecture only, use the following command:
# docker buildx build --platform linux/arm64 -t your_image_tag .
FROM --platform=linux/amd64 python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
# Initiating model_evaluation folder (if it doesn't already exist)
RUN mkdir -p model_evaluation
# install nltk dependency  
RUN [ "python3", "-c", "import nltk; nltk.download('punkt', download_dir='/usr/local/nltk_data')" ] 
# Set the entry point script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
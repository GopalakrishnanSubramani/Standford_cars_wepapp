FROM python:3.9-buster
RUN mkdir -p /app
COPY . /app
EXPOSE 8000
WORKDIR app
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r requirements.txt
CMD ["python3","app.py", "--host", "0.0.0.0", "--port", "5000"]


FROM ubuntu:latest

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y pip && \
    apt-get install -y xfce4 xorg

RUN pip install --upgrade pip && \
    pip install matplotlib && \
    pip install opencv-python && \
    pip install scipy

ENV DISPLAY host.docker.internal:0

CMD ["startxfce4"]
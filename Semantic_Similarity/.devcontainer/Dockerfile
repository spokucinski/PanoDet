FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install libcublas-12-0

RUN pip install --upgrade pip && \
    pip install pandas && \
    pip install gensim && \
    pip install matplotlib && \
    pip install seaborn && \
    pip install scikit-learn && \
    pip install scipy==1.10.1 && \
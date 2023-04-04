FROM tensorflow/tensorflow:latest

RUN apt-get update

RUN pip install --upgrade pip
RUN python -m pip install flwr
RUN python -m pip install pandas
RUN mkdir /FedLearn




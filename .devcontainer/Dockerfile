FROM ubuntu:bionic
MAINTAINER Apaar Shanker <apaar92@gmail.com>
USER root

RUN apt-get update
RUN apt-get install -y build-essential apt-utils
RUN apt-get install -y git cmake wget htop \ 
                       libssl-dev libffi-dev \
                       python3-pip python3-dev 

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
CMD [ "/bin/bash" ]

FROM ubuntu:16.04

LABEL maintainer="Arthur Jiang <arthursjiang@gmail.com>" \
name="PyTorch Tutorial" \
version="0.1"

# install python 3.6
RUN apt-get update && apt-get install -y software-properties-common curl && add-apt-repository ppa:deadsnakes/ppa \
&& apt-get update && apt-get install -y python3.6 python3.6-dev && apt-get clean \
&& cd /usr/bin; rm python python2 python2.7 python3 python3.5 python3.5m python3m; ln -s python3.6 python \
&& curl https://bootstrap.pypa.io/get-pip.py | python

# install py pkg
RUN pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl torchvision numpy jupyter

EXPOSE 8888

WORKDIR /notebooks

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", \
"--allow-root", "--ip=0.0.0.0", "--NotebookApp.token="]
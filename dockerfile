FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.11.9

WORKDIR /aiserver

COPY requirements.txt .
RUN pip install  -r requirements.txt

COPY . .

CMD ["python", "sever_async.py"]

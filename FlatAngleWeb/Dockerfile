FROM python:3.6
WORKDIR /Project/FlatAngleAPI

COPY requirements.txt ./
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

ENV LANG C.UTF-8

CMD ["gunicorn", "start:app", "-c", "./gunicorn.conf.py"]
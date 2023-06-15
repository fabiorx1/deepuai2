# 


FROM python:3.8.10

# 


WORKDIR /app

# 


COPY ./src /app/src
COPY ./requirements.txt /app/requirements.txt
RUN mkdir data
RUN mkdir data/applications
RUN mkdir data/datasets

# 


RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# 


CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]
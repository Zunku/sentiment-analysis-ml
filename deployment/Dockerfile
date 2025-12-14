FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app

# docker build -t sentiment-analysis . To build Docker image
# docker run -p 80:80 sentiment-analysis To run the Docker container
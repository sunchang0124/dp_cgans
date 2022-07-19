FROM python:3.9

WORKDIR /app

RUN apt-get update && apt-get upgrade -y

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

ENTRYPOINT [ "tail", "-f", "/dev/null" ]
FROM python:3.9

# This docker image is only used for debugging

WORKDIR /app

RUN apt-get update && apt-get upgrade -y

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
ENV PATH="${PATH}:/root/.poetry/bin"

ADD . .

RUN poetry install

ENTRYPOINT [ "tail", "-f", "/dev/null" ]
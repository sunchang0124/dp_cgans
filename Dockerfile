FROM python:3.9

# This docker image is only used for debugging

WORKDIR /app

RUN apt-get update && apt-get upgrade -y

RUN curl -sSL https://install.python-poetry.org | python -

ENV PATH="${PATH}:/root/.poetry/bin"

ADD . .

RUN poetry install


ENTRYPOINT [ "bash" ]

# Or leave the container hanging so we can connect to it with
# ENTRYPOINT [ "tail", "-f", "/dev/null" ]
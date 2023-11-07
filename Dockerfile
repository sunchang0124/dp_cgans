FROM python:3.10

# This docker image is only used for debugging

WORKDIR /app

RUN apt-get update && \
    apt-get upgrade -y && \
    pip install --upgrade pip

ADD . .

RUN pip install -e .


ENTRYPOINT [ "bash" ]

# Or leave the container hanging so we can connect to it with
# ENTRYPOINT [ "tail", "-f", "/dev/null" ]
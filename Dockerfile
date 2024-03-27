# base env: py39 ubuntu20.04
# 3.9 is needed for typing related stuff
FROM python:3.9-slim-buster

# install git
RUN apt-get update && apt-get install -y git

# upgrade to latest pip
RUN pip install --upgrade pip

COPY . /evoeval

RUN cd /evoeval && ls -l && pip install .

WORKDIR /app

ENTRYPOINT ["python3", "-m", "evoeval.evaluate"]

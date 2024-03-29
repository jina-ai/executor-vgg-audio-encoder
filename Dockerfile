
FROM jinaai/jina:3-py37-perf

# install git
RUN apt-get -y update && apt-get install -y curl && apt-get install -y libsndfile-dev

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# setup the workspace
COPY ./ /workspace
WORKDIR /workspace


ENV PYTHONPATH=/workspace
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
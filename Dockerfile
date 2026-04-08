FROM apache/airflow:2.11.0-python3.11
WORKDIR /app

USER root

# Install docker CLI for DockerOperator
RUN curl -o docker.tgz "https://download.docker.com/linux/static/stable/x86_64/docker-25.0.3.tgz" && \
    tar -xzf docker.tgz && \
    mv docker/docker /usr/local/bin/ && \
    rm -rf docker docker.tgz

# Add airflow user to docker group to access the socket if needed
RUN groupadd -f -g 999 docker && usermod -aG docker airflow

USER airflow

COPY requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt

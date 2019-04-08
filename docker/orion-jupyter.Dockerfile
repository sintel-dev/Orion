FROM python:3.6

EXPOSE 8888

RUN mkdir /app
COPY setup.py /app
RUN pip install -e /app && pip install jupyter

COPY orion /app/orion
COPY notebooks /app/notebooks
RUN mkdir -p /app/orion/notebooks/data \
    && cd /app/orion/notebooks/data \
    && wget https://d3-ai-orion.s3.amazonaws.com/S-1.csv \
    && wget https://d3-ai-orion.s3.amazonaws.com/S-2.csv \
    && wget https://d3-ai-orion.s3.amazonaws.com/E-1.csv \
    && wget https://d3-ai-orion.s3.amazonaws.com/P-1.csv \
    && mkdir -p /app/orion/data \
    && cd /app/orion/data \
    && wget https://d3-ai-orion.s3.amazonaws.com/anomalies.csv \
    && cp /app/orion/notebooks/data/* .

RUN adduser jupyter --uid 1000 --disabled-password --system && chown -R jupyter /app

WORKDIR /app
USER jupyter
CMD /usr/local/bin/jupyter notebook --ip 0.0.0.0 --NotebookApp.token=''

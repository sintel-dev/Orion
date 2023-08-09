FROM python:3.7

EXPOSE 8888

RUN mkdir /app
COPY setup.py /app
RUN pip install -e /app && pip install jupyter

COPY orion /app/orion
COPY tutorials /app/tutorials
RUN mkdir -p /app/orion/tutorials/data \
    && cd /app/orion/tutorials/data \
    && wget https://sintel-orion.s3.amazonaws.com/S-1.csv \
    && wget https://sintel-orion.s3.amazonaws.com/S-2.csv \
    && wget https://sintel-orion.s3.amazonaws.com/E-1.csv \
    && wget https://sintel-orion.s3.amazonaws.com/P-1.csv \
    && mkdir -p /app/orion/data \
    && cd /app/orion/data \
    && wget https://sintel-orion.s3.amazonaws.com/anomalies.csv \
    && cp /app/orion/tutorials/data/* .

RUN adduser jupyter --uid 1000 --disabled-password --system && chown -R jupyter /app

WORKDIR /app
USER jupyter
CMD /usr/local/bin/jupyter notebook --ip 0.0.0.0 --NotebookApp.token=''

FROM python:3.7.7

ENV work=/home/mcts_models

WORKDIR ${work}

COPY requirements.txt ${work}/
RUN pip install -r requirements.txt

COPY *.py ${work}/
COPY *.sh ${work}/
COPY datasets/ ${work}/datasets
COPY mf_tree/ ${work}/mf_tree
COPY configuration/ ${work}/configuration

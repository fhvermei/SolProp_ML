FROM continuumio/miniconda3:4.8.3-alpine

USER root

RUN apk update && apk upgrade && apk add bash

RUN /usr/sbin/addgroup -S askcos && \
    /usr/sbin/adduser -D -u 1000 askcos -G askcos && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/askcos/.profile && \
    echo "conda activate base" >> /home/askcos/.profile

RUN /opt/conda/bin/conda install -y chemprop_solvation coolprop descriptastorus rdkit=2020.09.5 pytorch-cpu=1.8 openjdk=11 torchserve -c conda-forge -c fhvermei -c pytorch -c rmg && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

COPY torchserve/config.properties /home/askcos/config.properties

EXPOSE 8080 8081

USER askcos

WORKDIR /home/askcos

COPY --chown=askcos:askcos ./solvation_predictor /home/askcos/solprop/solvation_predictor
COPY --chown=askcos:askcos ./torchserve/solprop.mar /home/askcos/model_store/solprop.mar

ENV PYTHONPATH=/home/askcos/solprop

ENV PATH=/opt/conda/bin:${PATH}

ENTRYPOINT ["torchserve", "--start", "--foreground", "--ts-config", "/home/askcos/config.properties"]
CMD ["--models", "all"]

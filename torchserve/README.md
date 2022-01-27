# Deploying solprop_ml using Torchserve

## Preface
Torchserve is the production-ready serving utility for pytorch models.

Ideally, Torchserve should be configured to deploy individual ML models, with the option to also create workflows involving multiple models.
However, the current approach here is to treat the entirety of solprop_ml as a black-box model, despite the fact that it involves multiple ensembles of multiple models.
While the current configuration is functional and requires minimal changes to solprop_ml, it is highly inefficient because the model predictions are run in sequence despite being parallelizable.

## Step 1: Build model archive

The model archive file can be easily created using the `create_mar.sh` script.
Note that a dummy .pt file is used since the actual models are loaded directly by solprop_ml.

```shell
bash create_mar.sh
```

## Step 2: Build Docker image

The included Dockerfile builds an image based on miniconda which installs all the necessary dependencies including this repo.

From the base directory:

```shell
docker build -t solprop:1.0 -f torchserve/Dockerfile .
```

From the torchserve directory:

```shell
docker build -t solprop:1.0 -f Dockerfile ..
```

## Step 3: Run Docker image
The Docker image can be run on its own for a standalone Torchserve server.
Note that the base configuration does not include SSL, so it may not be suitable for production on its own.

```shell
docker run -d -p 8080:8080 -p 8081:8081 solprop:1.0
```

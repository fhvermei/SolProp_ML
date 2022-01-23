#!/usr/bin/env bash

# Exit immediately on error
set -e

# Create a torchserve model archive file
torch-model-archiver \
  --model-name solprop \
  --version 1.0 \
  --serialized-file dummy.pt \
  --handler handler.py \
  --extra-files utils.py

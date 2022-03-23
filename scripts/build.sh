#!/bin/bash

set -e
docker build -t qhduan/onnx-paraphrase-multilingual-mpnet-base-v2 .
docker push qhduan/onnx-paraphrase-multilingual-mpnet-base-v2

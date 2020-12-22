## Model deployment

### flask + gunicorn (easiest, not for scaling)
https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166
https://www.datacamp.com/community/tutorials/machine-learning-models-api-python
`PYTHONPATH=. venv/bin/gunicorn -w 3 -t 600 --bind 192.168.0.215:4025 server:app`

### fastai + aws sagemaker 
https://github.com/fastai/course-v3/blob/master/docs/deployment_amzn_sagemaker.md

### fastai + torchserve + sagemaker
https://github.com/aws-samples/amazon-sagemaker-endpoint-deployment-of-fastai-model-with-torchserve

### fastai 1 + bentoml and kubernetes
https://course19.fast.ai/deployment_docker_kubernetes.html

### fastai2 + bentoml
https://docs.bentoml.org/en/latest/frameworks.html#fastai-v2

https://github.com/bentoml/gallery#fastai

### bentoml basics
https://docs.bentoml.org/en/latest/concepts.html?fbclid=IwAR3J05Bl7o5YLOF76v_WEIq1aAAgE0H0JJAphOr10VYuqf1qhfd0UKUIbs0
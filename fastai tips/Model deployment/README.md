## Model deployment

### fastai + pure pytorch deployment options (with jit tracing) examples:
https://github.com/maxmatical/fast.ai/blob/master/fastai_%2B_blurr_%2B_deberta_classification.ipynb

- can also add quantization
- with pytorch 1.9+, can use `with torch.inference_mode():` instead of `with torch.no_grad():` for even faster inference speedup


### Some fastai/pytorch deployment options:
https://twitter.com/TheZachMueller/status/1382459910907162626

includes: all fastai, pytorch, and ONNX

https://muellerzr.github.io/fastinference/

### model compression

- quantization, available in pytorch
- pruning, knowledge distillation (teacher student), lottery ticket hypothesis: https://nathanhubens.github.io/fasterai/
  - note: knowledge distillation will work for self-distillation, but not when there's unlabelled data


### flask + gunicorn (not the fastest, most general)

flask: good for web dev

gunicorn: helps solve production issues flask has (makes flask more scalable)

https://medium.com/technonerds/a-production-grade-machine-learning-api-using-flask-gunicorn-nginx-and-docker-part-1-49927238befb

https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166

https://www.datacamp.com/community/tutorials/machine-learning-models-api-python

sample bash command for running flask app (`main.py`) with gunicorn:
```
PYTHONPATH=. venv/bin/gunicorn -w 3 -t 600 --bind 0.0.0.0:{port_number} main:app

# -w is number of workers
# -t is timeout in ms
```
**some optimiations for flask with request batching:** https://www.sicara.ai/blog/optimize-response-time-api
- still not as fast as tfserve/torchserve, but useful to understand how model serving frameworks work to reduce latency 

### fastai + aws sagemaker 
https://github.com/fastai/course-v3/blob/master/docs/deployment_amzn_sagemaker.md

### fastai + torchserve + sagemaker
https://github.com/aws-samples/amazon-sagemaker-endpoint-deployment-of-fastai-model-with-torchserve

### BentoML (pure model serving)

### fastai 1 + bentoml and kubernetes
https://course19.fast.ai/deployment_docker_kubernetes.html

### fastai2 + bentoml
https://docs.bentoml.org/en/latest/frameworks.html#fastai-v2

https://github.com/bentoml/gallery#fastai

### bentoml basics
https://docs.bentoml.org/en/latest/concepts.html?fbclid=IwAR3J05Bl7o5YLOF76v_WEIq1aAAgE0H0JJAphOr10VYuqf1qhfd0UKUIbs0

### TensorRT (for deployment with Nvidia GPUS)
https://github.com/NVIDIA-AI-IOT/torch2trt

### Applying GPU to Snap-Scale Machine Learning Inference
https://eng.snap.com/applying_gpu_to_snap

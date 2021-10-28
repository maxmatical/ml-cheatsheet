## Model deployment

### Pytorch serialization best practices
1. have a `nn.Module`, `the_model = TheModel(*args)`
2. after training `torch.save(the_model.state_dict(), PATH)`
3. during inference:
  ```
  the_model = TheModelClass(*args, **kwargs)
  the_model.load_state_dict(torch.load(PATH))
  ```
4. call `the_model(x_inf)`

### Monitoring data drift 
1. For prediction models, want to measure confidence (i.e. is model confidence decreasing over time? Can indicate something)
2. Data side, can utilize some statistical measures on the new data coming in (eg cosine similarity on text embeddings between new data and old data, or maybe clustering to see if new data are being grouped apart from training data)
  - Not sure about validity of statistical measures, can be finicky 
3. Train a model to classify new vs training data (can the model detect differences in the data?)
4. every once in a while, you can grab a sample of the new data, and manually validate the model performance, and see if model predictions are diverging from "ground truth", especially over time


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

### TensorRT (for deployment with Nvidia GPUS)
https://github.com/NVIDIA-AI-IOT/torch2trt

### Applying GPU to Snap-Scale Machine Learning Inference
https://eng.snap.com/applying_gpu_to_snap

### Scaling BERT to 1B+ daily request on CPUs
https://blog.roblox.com/2020/05/scaled-bert-serve-1-billion-daily-requests-cpus/

Key takeaways:

1. GPU works better for batching, but **real time requests worked better for non batching**, use CPU
2. set `num_threads` to 1
3. use smaller model (distillbert, distillroberta etc.)
4. dynamic shapes (turn `padding=False` when tokenizing)
  - works because using bs of 1
5. quantize
6. cache respones to common text inputs (via token ids)
7. favor horizontal scaling over vertical


## Deployment Frameworks

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

### Flask + gevent (better than gunicorn)
- adds async to flask
- https://www.google.com/search?client=firefox-b-d&q=why+use+gevent+with+flask

### Fastapi
- async by default
- https://fastapi.tiangolo.com/

### fastai + aws sagemaker 
https://github.com/fastai/course-v3/blob/master/docs/deployment_amzn_sagemaker.md

### fastai + torchserve + sagemaker
https://github.com/aws-samples/amazon-sagemaker-endpoint-deployment-of-fastai-model-with-torchserve

## Pure model serving

### BentoML (pure model serving)

### fastai 1 + bentoml and kubernetes
https://course19.fast.ai/deployment_docker_kubernetes.html

### fastai2 + bentoml
https://docs.bentoml.org/en/latest/frameworks.html#fastai-v2

https://github.com/bentoml/gallery#fastai

### bentoml basics
https://docs.bentoml.org/en/latest/concepts.html?fbclid=IwAR3J05Bl7o5YLOF76v_WEIq1aAAgE0H0JJAphOr10VYuqf1qhfd0UKUIbs0


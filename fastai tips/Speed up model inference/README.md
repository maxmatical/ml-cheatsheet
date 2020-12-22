## speed up model inference
### Speed up inference with jit and quantization
[jit + quantization](https://forums.fast.ai/t/using-torch-quantization/56582)
 - dynamic quantization: https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/dynamic_quantization_bert_tutorial.ipynb#scrollTo=IzyVSIKYIgN5
 - use try static quantization `torch.quantization.quantize`
 
quantize with
```
model = learn.model
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
learn.model = quantized_model
```
### Using Dynamic and static quantization
https://spell.ml/blog/pytorch-quantization-X8e7wBAAACIAHPhT

### Using jit to export `learn.model` as `torch.script`
https://drhb.github.io/blog/fastai/2020/03/22/Fastai-Jit.html

### speed up fastai inference
https://forums.fast.ai/t/speeding-up-fastai2-inference-and-a-few-things-learned/66179

- it looks like using `learn.model` directly instead of `learn.get_preds` gives some speedup

- for nlp (transformers), to tokenize and numericalze text, we need to do (from https://huggingface.co/transformers/custom_datasets.html?highlight=tokenizer%20encode) 

``` 
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
i2f = IntToFloatTensor()
texts = [some list of texts]
test_encodings = tokenizer(test_texts, truncation=True, padding=True)
# test_encodings will look like {'input_ids': [[0, 42891, 6, 127, 766, 16, 19220, 2], [0, 8396, 662, 385, 1584, 571, 2]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]}

# inference loop for list of texts:
batches = []
input_ids_batches = []
attention_mask_batches = []
outs = []
inps = []
k = 0
for input_ids, attention_mask in zip(test_encodings["input_ids"], test_encodings["attention_mask"]):
  input_ids_batch.append(ToTensor(input_ids))
  attention_mask_batch.append(ToTensor(attention_mask))
  k +=1 
  if (k+1)%50 == 0 or k == len(texts):
    input_ids_batches.append(torch.cat([b for b in input_ids_batch]))
    attention_mask_batches.append(torch.cat([b for b in attention_mask_batch]))
    # alternatively if need to convert to float tensors, use 
    # input_ids_batches.append(torch.cat([i2f(b) for b in batch]))
    # attention_mask_batches.append(torch.cat([i2f(b) for b in attention_mask_batch]))
    input_ids_batch = []
    attention_mask_batch = []
    
learner.model.eval()
with torch.no_grad():
    for ids, mask in zip(input_ids_batches, attention_mask_batches):
        outs.append(learner.model(ids, mask))
# some decoding here
```

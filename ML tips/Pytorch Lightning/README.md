# Useful Pytorch Lightning Resources

[With Huggingface](https://github.com/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb)

[Pytorch Lightning kaggle kernel with WandB (examples with callbacks)](https://www.kaggle.com/ayuraj/use-pytorch-lightning-with-weights-and-biases)

[Transfer learning with Resnet34](https://www.kaggle.com/ymicky/pytorch-lightning-resnet34-baseline)

[Kaggle 1st place notebooks (with some best practices)](https://devblog.pytorchlightning.ai/3-pytorch-lightning-winning-community-kernels-to-inspire-your-next-kaggle-victory-ea355456229a)

## Pytorch lightning bolts
https://pytorch-lightning-bolts.readthedocs.io/en/0.3.0/introduction_guide.html

- useful for self-supervised learning stuff

## Useful examples:

ray lighting: https://medium.com/pytorch/getting-started-with-ray-lightning-easy-multi-node-pytorch-lightning-training-e639031aff8b
  - multi-node training
  - contains nice template for pytorch lightning module training loop


## logging metrics
```
def training_step(self, batch, batch_idx):
    ...
    self.log("train_acc", acc, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

def validation_step(self, batch, batch_idx):
    ...
    self.log("val_acc", acc, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
```

- `LearningRateMonitor`: `lr_monitor = LearningRateMonitor(logging_interval='step')`




## Useful fastai functionalities in ptl

### lr finder

lr finder: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#auto-lr-find

[lr finder example](https://lightning-flash.readthedocs.io/en/latest/notebooks/flash_tutorials/electricity_forecasting.html)
```
res = trainer.tuner.lr_find(model, datamodule=datamodule, min_lr=1e-5)
print(f"Suggested learning rate: {res.suggestion()}")
res.plot(show=True, suggest=True).show()

model.learning_rate = res.suggestion()
```


### LR schedules

lr schedules: use pytorch lr shcedules like `OneCycleLR` in the training step, see here: https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html#learning-rate-scheduling-manual

see: https://github.com/mgrankin/over9000/blob/master/train.py

`flat_cos`:
```
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

class ConcatLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, scheduler1, scheduler2, total_steps, pct_start=0.5, last_epoch=-1):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.step_start = float(pct_start * total_steps) - 1
        super(ConcatLR, self).__init__(optimizer, last_epoch)
    
    def step(self):
        if self.last_epoch <= self.step_start:
            self.scheduler1.step()
        else:
            self.scheduler2.step()
        super().step()
        
    def get_lr(self):
        if self.last_epoch <= self.step_start:
            return self.scheduler1.get_lr()
        else:
            return self.scheduler2.get_lr()
            
            
class LightningModel(pl.LightningModule):
    ...
    
    def configure_optimizers(self):
        optimizer = self.opt_func(self.parameters(), lr=self.lr)
        if self.sched_type == 'flat_and_anneal':
            dummy = LambdaLR(optimizer, d)
            cosine = CosineAnnealingLR(optimizer, self.total_steps*(1-self.ann_start))
            scheduler = ConcatLR(optimizer, dummy, cosine, self.total_steps, self.ann_start)
        else:
            scheduler = OneCycleLR(optimizer, max_lr=self.lr, total_steps=self.total_steps, pct_start=0.3,
                                                            div_factor=10, cycle_momentum=True)        
        meta_sched = {
         'scheduler': scheduler,
         'interval': 'step',
         'frequency': 1
        }  
        return [optimizer], [meta_sched]
```

- `ReduceLROnPlateau`: use pytorch implementation, eg
```
def __init__(self):
    super().__init__()
    self.automatic_optimization = False


def training_epoch_end(self, outputs):
    sch = self.lr_schedulers()

    # If the selected scheduler is a ReduceLROnPlateau scheduler.
    if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
        sch.step(self.trainer.callback_metrics["loss"])
```


### LR schedules with multi gpu training
see: https://github.com/PyTorchLightning/pytorch-lightning/discussions/2149

example
```
class DeepLabv3ResNet101(pl.LightningModule):
    # ...
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(
            limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    @property
    def steps_per_epoch(self) -> int:
        return self.num_training_steps // self.trainer.max_epochs

    def configure_optimizers(self):
        # Get trainable params
        head_params = itertools.chain(self.model.classifier.parameters(),
                                      self.model.aux_classifier.parameters())
        backbone_params = itertools.chain(self.model.backbone.layer4.parameters(),
                                          self.model.backbone.layer3.parameters())

        # Init optimizer and scheduler
        optimizer = torch.optim.AdamW([
            {"params": head_params},
            {"params": backbone_params, "lr": self.hparams.backbone_lr},
        ], lr=self.hparams.lr, amsgrad=True, weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                           max_lr=[self.hparams.lr,
                                                                   self.hparams.backbone_lr],
                                                           total_steps=self.num_training_steps)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
```

### using metrics like accuracy

`accuracy` metric: `from torchmetrics.functional import accuracy`

### label smoothing
label smoothing: either use [fastai implementation](https://github.com/fastai/fastai/blob/e80adfc3786464b38c487a0382424c6197166499/fastai/losses.py#L13) or [see here](https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch) for pure pytorch

```
criterion = LabelSmoothingCrossEntropy()
```

discriminative learning rates: 

## Optimizers

### SAM with pytorch lightning
https://github.com/davda54/sam

when using SAM with PL and ddp, do something like
https://github.com/PyTorchLightning/pytorch-lightning/discussions/10792
```

```

## Ranger21
https://github.com/lessw2020/Ranger21

- no need for lr schedule (define lr schedule in the optimizer itself), just need to call `trainer.fit()`
- `use_madgrad = True` might be better for transformers

## scaling batch size
https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#auto-scaling-of-batch-size

## Gradient accumulation and clipping
https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#training-tricks

**Note**: for manual optimization, when `self.automatic_optimization = False`, must manually specify gradient accumulation, see [here](https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html#gradient-accumulation)

## Model parallelism (inc. deepspeed)
https://pytorch-lightning.readthedocs.io/en/latest/advanced/advanced_gpu.html

### tips for using `strategy="deepspeed_stage_3_offload"`
Here is some helpful information when setting up DeepSpeed ZeRO Stage 3 with Lightning.

If you’re using Adam or AdamW, ensure to use FusedAdam or DeepSpeedCPUAdam (for CPU Offloading) rather than the default torch optimizers as they come with large speed benefits

Treat your GPU/CPU memory as one large pool. In some cases, you may not want to offload certain things (like activations) to provide even more space to offload model parameters

When offloading to the CPU, make sure to **bump up the batch size** as GPU memory will be freed

We also support sharded checkpointing. By passing save_full_weights=False to the DeepSpeedPlugin, we’ll save shards of the model which allows you to save extremely large models. However to load the model and run test/validation/predict you must use the Trainer object.


## Mixup for pytorch lightning
https://github.com/PyTorchLightning/pytorch-lightning/issues/790
- helpful for vision, may be helpful for nlp as well (classification problems)
- For NLP: follow strategy from https://arxiv.org/abs/2010.02394
    - 2 separate datasets, for training, dataloader loads `x_i, x_j, y_i, y_j` and for validation/inference, dataloader only loads `x_i, y_i`

## FP16 with amp
`pl.Trainer(gpus=1, precision=16)`


 
 
 ## Deployment best practices
 1. manually define `nn.Module`, decouple from `pl.LightningModule`
 2. Create `pl.LightningModule` as a wrapper for `nn.Module` model
 3. Train using trainer
 4. save weights
 5. During inference, manually define `nn.Module` again, and load trained weights with 
    ```
    new_model.load_state_dict(torch.load(saved_model_pth))
    ```
    
## Stochastic weight averaging (SWA)
https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.callbacks.StochasticWeightAveraging.html

- could improve generalization by searching for flatter minima
- using SGD only? may not be useful for other optimizers like AdamW, Ranger etc.
  - used succesfully with Adam see https://twitter.com/jeankaddour/status/1494437438856572932
- **IMPOrTANT**: if using save model checkpoints, model at the end of each epoch (eg for checkpoints, train/val metrics, etc.) is NOT of the averaged model weights.
  - so if you train with or without swa (assuming same seed, same lr schedule etc.), the train/val metrics will be the same across the 2
  - so if you load best model checkpoint before saving model, you will not save the averaged model
  - to save the SWA model, only save the model (no loading checkpoints) after entire training is done

## Exponential Moving Average (EMA):
**potential EMA callback**: https://github.com/PyTorchLightning/pytorch-lightning/issues/10914
  - could be more useful that naive implementation

timm EMA callback with pytorch lightning callback: https://benihime91.github.io/gale/collections.callbacks.ema.html?fbclid=IwAR34fW167fiwn6Xm5L8F0aYYw4EvZn5gONiO5bw-M9sJ1PXc3KHQmY4w19k


### EMA/SWA with model checkpoints/early stopping

if using EMA/SWA, better to not use model checkpoints/early stopping, the process would be 

- can also be extended to general training
- can be done after HPO with a set number of epochs
1. set `n_epochs` and train model
2. if at the end of training, monitored metrics (eg `val_acc`) is still improving, increase `n_epochs` and train again
3. if at the end of training, monitored metrics is worse than in the middle of training (eg epoch 25), set `n_epochs` to somewhere around where metrics was the best (25)


## Trainer strategy api
https://devblog.pytorchlightning.ai/announcing-the-new-lightning-trainer-strategy-api-f70ad5f9857e
```
# train on 8 GPUs (same machine (ie: node))
trainer = Trainer(devices=8, accelerator="gpu", strategy="ddp", precision=16)

# train on 32 GPUs (4 nodes 8 gpu each node)
trainer = Trainer(devices=8, accelerator="gpu", num_nodes=4, strategy="ddp", precision=16)

# train on tpu with 8 cores
trainer = Trainer(devices=8, accelerator="tpu", strategy="ddp", precision=16)

```
**Note**: 
- always use `strategy="ddp"` with multi-device training. 
- for `device=1` use `strategy=None`
- use `strategy="ddp2"` In certain cases, it’s advantageous to use all batches on the same machine instead of a subset. For instance, you might want to compute a NCE loss where it pays to have more negative samples.

## distributed training with Bagua

https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu.html#bagua

- may provide speed up over ddp

## Dealing with numerical instability with mixed precision
- especially prevalent when scaling
- try using `"bf16"` instead of `16` for precision
  - improved numerical stability
  - useful for Ampere gpus (3090, A100 etc.)
  - requires pytorch `1.10.0` and up
- warning: if a model is pre-trained in `bf16`, fine-tuning in `fp16` will result in numerical instability
- bf16 has worse precision than fp16, sometimes you still want to run in fp16
- no longer need to manually scale loss. especially useful if running `self.automatic_optimization = False` with `SAM`, 
```
from torch.cuda.amp import autocast
with autocast(dtype=torch.bfloat16):
    loss, outputs = ...
    
```

## Cross validation with pytorch lightning
https://github.com/SkafteNicki/pl_cross

## speeding up pytorch lightning
https://william-falcon.medium.com/pytorch-lightning-vs-deepspeed-vs-fsdp-vs-ffcv-vs-e0d6b2a95719

- can be slower than other frameworks due to all the additional things 
- if want to speed up training, do the following
```
trainer = Trainer(
    ...
    enable_progress_bar=False,
    enable_model_summary=False,
    enable_checkpointing=False,
    logger=False,
    replace_sampler_ddp=False
)
```

## using pytorch lightning with composer
https://william-falcon.medium.com/pytorch-lightning-vs-deepspeed-vs-fsdp-vs-ffcv-vs-e0d6b2a95719

- composer gives lots of nice training tricks
- can use `composer.functional` in conjunction with pytorch/lightning

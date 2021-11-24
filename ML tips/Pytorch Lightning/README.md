# Useful Pytorch Lightning Resources

[With Huggingface](https://github.com/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb)

[Pytorch Lightning kaggle kernel with WandB (examples with callbacks)](https://www.kaggle.com/ayuraj/use-pytorch-lightning-with-weights-and-biases)

[Transfer learning with Resnet34](https://www.kaggle.com/ymicky/pytorch-lightning-resnet34-baseline)

[Kaggle 1st place notebooks (with some best practices)](https://devblog.pytorchlightning.ai/3-pytorch-lightning-winning-community-kernels-to-inspire-your-next-kaggle-victory-ea355456229a)

## Useful examples:

ray lighting: https://medium.com/pytorch/getting-started-with-ray-lightning-easy-multi-node-pytorch-lightning-training-e639031aff8b
  - multi-node training
  - contains nice template for pytorch lightning module training loop


## Useful callbacks
- [early stopping](https://pytorch-lightning.readthedocs.io/en/latest/common/early_stopping.html)
eg
```
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=6)

```

- [save best weights](https://pytorch-lightning.readthedocs.io/en/latest/common/weights_loading.html)
eg
```
checkpoint_callback = ModelCheckpoint(
  dirpath="checkpoints",
  filename="best-checkpoint",
  save_top_k=1,
  verbose=True,
  monitor="val_loss",
  mode="min"
)
```

**Note**: for early stopping and model checkpoint, if want `monitor="val_acc"` or similar metric, need to calculate metric in 

```
def training_step(self, batch, batch_idx):
    ...
    self.log("train_acc", acc, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

def validation_step(self, batch, batch_idx):
    ...
    self.log("val_acc", acc, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
```

- `LearningRateMonitor`: `lr_monitor = LearningRateMonitor(logging_interval='step')`

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



[stochastic weight averaging (SWA)](https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#stochastic-weight-averaging)

## Useful fastai functionalities in ptl

### lr finder

lr finder: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#auto-lr-find


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

## Mixup for pytorch lightning
https://github.com/PyTorchLightning/pytorch-lightning/issues/790
- helpful for vision, may be helpful for nlp as well (classification problems)
- For NLP: follow strategy from https://arxiv.org/abs/2010.02394
    - 2 separate datasets, for training, dataloader loads `x_i, x_j, y_i, y_j` and for validation/inference, dataloader only loads `x_i, y_i`

## FP16 with amp
`pl.Trainer(gpus=1, precision=16)`

## DDP with multiple gpus (and nodes(
```
# train on 8 GPUs (same machine (ie: node))
trainer = Trainer(gpus=8, accelerator="ddp")

# train on 32 GPUs (4 nodes)
trainer = Trainer(gpus=8, accelerator="ddp", num_nodes=4)
```
 **Note**: use `accelerator="ddp2"` In certain cases, it’s advantageous to use all batches on the same machine instead of a subset. For instance, you might want to compute a NCE loss where it pays to have more negative samples.
 
 
 ## Deployment best practices
 1. manually define `nn.Module`, decouple from `pl.LightningModule`
 2. Create `pl.LightningModule` as a wrapper for `nn.Module` model
 3. Train using trainer
 4. save weights
 5. During inference, manually define `nn.Module` again, and load trained weights with 
    ```
    new_model.load_state_dict(torch.load(saved_model_pth))
    ```
    
## Exponential Moving Average (EMA):
https://github.com/fadel/
- should be fairly simple with `self.automatic_topimization=False` in `LightningModule`
- just add `ema.update()` after `opt.step()` in `training_step` of `LightningModule`

use EMA in PL: https://forums.pytorchlightning.ai/t/adopting-exponential-moving-average-ema-for-pl-pipeline/488

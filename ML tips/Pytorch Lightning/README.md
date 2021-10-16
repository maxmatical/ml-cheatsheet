# Useful Pytorch Lightning Resources

[With Huggingface](https://github.com/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb)

[Pytorch Lightning kaggle kernel with WandB (examples with callbacks)](https://www.kaggle.com/ayuraj/use-pytorch-lightning-with-weights-and-biases)

[Transfer learning with Resnet34](https://www.kaggle.com/ymicky/pytorch-lightning-resnet34-baseline)

[Kaggle 1st place notebooks (with some best practices)](https://devblog.pytorchlightning.ai/3-pytorch-lightning-winning-community-kernels-to-inspire-your-next-kaggle-victory-ea355456229a)


## Useful callbacks
[early stopping](https://pytorch-lightning.readthedocs.io/en/latest/common/early_stopping.html)

[save best weights](https://pytorch-lightning.readthedocs.io/en/latest/common/weights_loading.html)

`ReduceLROnPlateau`: use pytorch implementation, eg
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

## Useful fastai functionalities in ptl
[lr finder]: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#auto-lr-find

`accuracy` metric: `from torchmetrics.functional import accuracy`

label smoothing: either use [fastai implementation](https://github.com/fastai/fastai/blob/e80adfc3786464b38c487a0382424c6197166499/fastai/losses.py#L13) or [see here](https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch) for pure pytorch

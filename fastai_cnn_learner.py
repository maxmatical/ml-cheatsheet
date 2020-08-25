# Import linraries

import os
import urllib
import shutil

from tqdm import tqdm
# from sklearn.metrics import confusion_matrix
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from optimizers.optimizers import Ranger
import torch
import warnings
warnings.filterwarnings('ignore')


def train_fit_fc():
    # train head
    learn.fit_fc(10,
                lr,
                start_pct=0.7,
                callbacks = [SaveModelCallback(learn, every='improvement', monitor='accuracy', 
                                                name='stage1')])
    # load stage1
    learn.load("stage1")
    learn.unfreeze()
    learn.fit_fc(40,
                lr/10, # try between [lr/10, lr/50]
                start_pct=0.7,
                callbacks = [SaveModelCallback(learn, every='improvement', monitor='accuracy', 
                                                name='stage2'),
                            EarlyStoppingCallback(learn, monitor="accuracy", patience = 10)])

    # further training                                
    learn.load("stage2")
    learn.export("/home/max/project/stage2-checkpt")

    learn.unfreeze()
    learn.fit_fc(60,
                lr/100, # try between [lr/50, lr/100]
                start_pct=0.7,
                callbacks = [SaveModelCallback(learn, every='improvement', monitor='accuracy', 
                                                name='stage3'),
                            EarlyStoppingCallback(learn, monitor="accuracy", patience = 15)])
    learn.load("stage3")
    learn.export("/home/max/project/stage3-checkpt")



def train_one_cycle():
    # train head
    learn.fit_one_cycle(10,
                        lr,
                        callbacks = [SaveModelCallback(learn, every='improvement', monitor='accuracy', 
                                                        name='stage1-onecycle')])

    # load stage1
    learn.load("stage1-onecycle")

    # training full model
    learn.unfreeze()
    learn.fit_one_cycle(10, 
                        max_lr=slice(lr/100,lr/2), # try between [lr/2, lr/10]
                        callbacks = [SaveModelCallback(learn, every='improvement', monitor='accuracy', 
                                                        name='stage2-onecycle')])

    # stage3 further training
    learn.load("stage2-onecycle")
    learn.export("/home/max/project/stage2-onecycle-checkpt")

    learn.unfreeze()
    learn.fit_one_cycle(15, 
                        max_lr=slice(lr/100,lr/100), # try between [lr/10, lr/100]
                        callbacks = [SaveModelCallback(learn, every='improvement', monitor='accuracy', 
                                                        name='stage3-onecycle')])

    # exporting models
    learn.load("stage3-onecycle")
    learn.export("/home/max/project/stage3-onecycle-checkpt")

                                        

if __name__ == "__main__":
    torch.cuda.set_device(0)

    # define image data directory path
    DATA_DIR='./occasion_data'


    # The directory under the path is the label name.
    os.listdir(f'{DATA_DIR}')

    torch.cuda.is_available()
    tfms = get_transforms(max_rotate=20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4,
                          p_affine=1., p_lighting=1.)

    # create image data bunch
    data = ImageDataBunch.from_folder(DATA_DIR, 
                                      train=".", 
                                      valid_pct=0.2,
                                      ds_tfms=tfms,
                                      size=224,
                                      padding_mode = "reflection",
                                      bs=64, 
                                      num_workers=0).normalize(imagenet_stats)

    # check classes
    print(f'Classes: \n {data.classes}')

    # build model (use resnet34)
    learn = cnn_learner(data, 
                        models.resnet50, 
                        metrics=accuracy, 
                        model_dir="/home/max/gender_classification/model/",
                        loss_func=LabelSmoothingCrossEntropy(),
                        opt_func = partial(Ranger),
                        true_wd = True,
                        bn_wd = False)
    learn.mixup()

    # find lr
    # can comment this out in later runs
    learn.lr_find()
    lr_find = learn.recorder.plot(suggestion = True, skip_end=15, return_fig = True)
    lr_find.savefig("lr_find.png")

    lr = 3e-2
    
    learn.to_fp16()

    # gen_data()
    train_fit_fc()
    # train_one_cycle()
import numpy as np
import glob
import os
import torch
from torch.utils.tensorboard import SummaryWriter

model_path = 'exp/experiments_nbs_10_randtopk_25_large/scannet_nbs10_s0.25_p192_rnn_vgg16unet3_gruunet4.64.3'

writer = SummaryWriter(os.path.join(model_path, 'tb'))
log_ = glob.glob(os.path.join(model_path, "*.log"))[0]

scannet_eval_sets = [
    "scene0707_00", "scene0709_00",  
    "scene0711_00", "scene0713_00",  
    "scene0715_00", "scene0708_00",  
    "scene0710_00", "scene0712_00",  
    "scene0714_00", "scene0716_00"
]

ll = ["loss/train_loss"]
for x in scannet_eval_sets:
    ll.append(f"loss/{x}_loss")

layout = {
    "abc": {
        "loss": ["Multiline", ll],
    },
}
writer.add_custom_scalars(layout)

f = open(log_, 'r')
for line in f.readlines():
    if ' train ' in line and 'loss=' in line:
        iter_ = int(line.split('train')[1].split('/')[0].strip())
        loss = float(line.split('loss=')[1].split('=')[0].strip())
        writer.add_scalar('loss/train_loss', loss, global_step=iter_, walltime=None)
    elif 'Evaluating' in line:
        eval_scene = line.split('set')[1].strip()
    elif 'Eval iter' in line:
        eval_iter = int(line.split('iter')[1].strip())
    elif 'eval_loss=' in line:
        loss = float(line.split('eval_loss=')[1].split('=')[0].strip())
        writer.add_scalar(f'loss/{eval_scene}_loss', loss, global_step=eval_iter, walltime=None)
writer.close()

f.close()

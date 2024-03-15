import torch
from torch import optim
from models.unet import UNet
from datasets.octa_500 import get_sets
from framework import Framework
from tools.evaluator import Eval_Seg
from tools.modules import Logger
from tools.loss_func import DiceLoss
import os


os.environ['CUDA_VISIBLE_DEVICES'] = "2"

if __name__ == "__main__":    
    model = UNet(n_channels=1, n_classes=1)
    config = {
        'mission': 'U-Net',
        'data_set': 'OCTA_500',
        'batch_size': 4,
        'max_epoch': 100,
        
        'optmizer': 'Adam',
        'Adam': {
            'lr': 1e-4,
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'weight_decay': 1e-4,
        },
        
        'scheduler': 'StepLR',
        'StepLR': {
            'step_size': 30,
            'gamma': 1,
        },

        'loss_func': 'DiceLoss',
    }

    optimizer = optim.Adam(model.parameters(), lr=config['Adam']['lr'], betas=config['Adam']['betas'], eps=config['Adam']['eps'], weight_decay=config['Adam']['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['StepLR']['step_size'], gamma=config['StepLR']['gamma'])
    loss_func = DiceLoss()
    evaluator = Eval_Seg()
    logger = Logger(mission = config['mission'],
                    log_layers = ['inc'])
    components = {
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_func': loss_func,
        'evaluator': evaluator,
        'logger': logger,
    }

    train_set, val_set, test_set = get_sets()
    seg = Framework(model=model, 
                    config=config,
                    components=components,
                    train_set=train_set, 
                    val_set=val_set,
                    test_set=test_set)
    
    seg.set_device('cuda')
    seg.train()
    seg.test()
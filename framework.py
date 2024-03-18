from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch

import os
from tqdm import tqdm
import numpy as np
from tools.metrics import thresholding
from tools.func_tools import *
class Framework(object):
    def __init__(self,
                 model:nn.Module, 
                 config:dict,
                 train_set:Dataset, 
                 test_set:Dataset,
                 components:dict,
                 val_set:Dataset=None,
                 device=torch.device('cuda'),
                 seed=None) -> None:
        
        self.mission = config['mission']
        self.model = model

        self.train_set = train_set
        self.test_set = test_set
        if val_set is None:
            val_set = test_set
        self.val_set = val_set
        
        self.seed = seed
        self.device = device
        self.config = config
        self.epoch = 0
                


        # Here is Configurations
        self.train_tools =  components
        self.loss_fn =      components['loss_func']
        self.optimizer =    components['optimizer']
        self.lr_scd =       components['scheduler']
        self.logger =       components['logger']
        self.evaluator =    components['evaluator']
        self.judger =       components['judger']
        if seed is not None:
            self.set_seed(seed)
    
    def set_device(self, device) -> None:
        self.model = self.model.to(device)
        self.device = device


    
    def set_seed(self, seed:int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # Training Part
    def train_one_epoch(self, dataloader) -> torch.tensor:
        self.model.train()
        loss_sum = 0
        with tqdm(total=len(dataloader), desc=f'Training', position=1, leave=False, colour='green', ncols=120) as _tqdm:
            for batch_idx, (X, Y) in enumerate(dataloader):
                self.optimizer.zero_grad()
                X, Y = X.to(self.device), Y.to(self.device)
                pred = self.model(X)
                loss = self.loss_fn(pred, Y)
                loss_sum += loss
                loss.backward()
                self.optimizer.step()
                _tqdm.set_postfix_str(f"Train Loss: {loss:.4f}")
                _tqdm.update(1)
        self.epoch += 1
        return loss_sum / len(dataloader)


    # Validation Part
    def val_one_epoch(self, dataloader) -> torch.tensor:
        self.model.eval()
        loss_sum = 0
        with tqdm(total=len(dataloader), desc=f'Validation', position=1, leave=False, colour='green', ncols=120) as _tqdm:
            for batch_idx, (X, Y) in (enumerate(dataloader)):
                X, Y = X.to(self.device), Y.to(self.device)
                pred = self.model(X)
                loss = self.loss_fn(pred, Y)
                loss_sum += loss
                _tqdm.set_postfix_str(f"Validation Loss: {loss:.4f}")
        return loss_sum / len(dataloader)

    def val_one_epoch_metric(self, dataloader):
        self.model.eval()
        self.evaluator.start()
        loss_sum = 0
        with tqdm(total=len(dataloader), desc=f'Validation', position=1, leave=False, colour='green', ncols=120) as _tqdm:
            with torch.no_grad():
                for batch_idx, (X, Y) in (enumerate(dataloader)):
                    X, Y = X.to(self.device), Y.to(self.device)
                    pred_Y = self.model(X)
                    loss = self.loss_fn(pred_Y, Y)
                    pred_Y = pred_Y.detach().cpu().numpy()
                    Y = Y.detach().cpu().numpy()

                    threshed_Y = thresholding(pred_Y, threshold=0.5)

                    self.evaluator(threshed_Y, Y)
                    loss_sum += loss
                    _tqdm.update(1)
                    _tqdm.set_postfix_str(f"Validation Loss: {loss:.4f}")
        return loss_sum / len(dataloader), self.evaluator.stop()


    # Testing Part
    def test_one_epoch(self, dataloader):
        self.model.eval()
        self.evaluator.start()
        loss_sum = 0

        with tqdm(total=len(dataloader), desc=f'Testing', position=0, colour='green', ncols=120) as _tqdm:
            with torch.no_grad():
                for batch_idx, (X, Y) in (enumerate(dataloader)):
                    X, Y = X.to(self.device), Y.to(self.device)
                    pred_Y = self.model(X)
                    loss = self.loss_fn(pred_Y, Y)
                    pred_Y = pred_Y.detach().cpu().numpy()
                    Y = Y.detach().cpu().numpy()

                    threshed_Y = thresholding(pred_Y, threshold=0.5)

                    self.evaluator(threshed_Y, Y)
                    loss_sum += loss
                    _tqdm.update(1)

        return loss_sum / len(dataloader), self.evaluator.stop()


    def train(self) -> None:
        self.logger.clean()
        batch_size = self.config['batch_size']
        max_epoch = self.config['max_epoch']
  
        train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=True)

        # Write your own trainning Code Here
        with tqdm(total=max_epoch, position=0, leave=False, colour='red', ncols=120) as _tqdm:
            _tqdm.update(self.epoch)
            for epoch in range(self.epoch, max_epoch):
                _tqdm.set_description_str(f"Epoch {epoch+1}/{max_epoch}")
                train_loss = self.train_one_epoch(train_loader)
                val_loss, scores = self.val_one_epoch_metric(val_loader)
                best = self.judger.update(1 - val_loss)
                self.logger.log_loss(train_loss, val_loss, epoch+1)
                self.logger.log_score(scores, epoch+1)
                self.logger.log_grad(self.model, epoch+1)
                self.lr_scd.step()
                _tqdm.set_postfix_str(f"Best: {0:.4f}, Latest: {val_loss:.4f}")
                _tqdm.update(1)
                self.save(epoch, f"Latest")
                if best:
                    self.save(epoch, f"Best")
        
    def resume_train(self, postfix) -> None:
        path = f"checkpoints/{self.mission}/CKPT_{postfix}.pth"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model \"CKPT_{postfix}\" not found")

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scd.load_state_dict(checkpoint['scheduler'])
        conf = checkpoint['config']
        conf.pop('max_epoch')
        conf.pop('batch_size')
        self.config.update(conf)
        self.epoch = checkpoint['epoch'] + 1
        print("Start from epoch", self.epoch+1)
        self.train()

    def test(self) -> None:
        test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False)
        test_loss, scores = self.test_one_epoch(test_loader)
        print_dict(scores)

    def save(self, epoch, postfix) -> None:
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.lr_scd.state_dict(),
            'config': self.config,
            'epoch': epoch,
        }
        folder = f'checkpoints/{self.mission}/'
        if not os.path.exists(folder):
            os.mkdir(folder)
        path = f"{folder}/CKPT_{postfix}.pth"
        torch.save(checkpoint, path)
        # print(f"\r Model \"CKPT_{postfix}\" saved", end='')
    
    def load(self, postfix) -> None:
        path = f"checkpoints/{self.mission}/CKPT_{postfix}.pth"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model \"CKPT_{postfix}\" not found")

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scd.load_state_dict(checkpoint['scheduler'])
        self.config = checkpoint['config']

        print(f"Model \"CKPT_{postfix}\" loaded")

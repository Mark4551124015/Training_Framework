import os
from tensorboardX import SummaryWriter


class Logger(object):
    def __init__(self, mission):
        path = r"logs/tensorboard-log"
        self.writer = SummaryWriter(path)
        self.mission = mission


    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        

    def log_loss(self, train_loss, val_loss, step):
        loss = {'train': train_loss, 'val': val_loss}
        self.writer.add_scalars(f'{self.mission}/Loss', loss, step)
    
    def log_score(self, scores:dict, step:int):
        self.writer.add_scalars(f'{self.mission}/Metric', scores, step)

    def log_grad(self, model, step):
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.writer.add_histogram(tag, value.data.cpu().numpy(), step)
            self.writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), step)
            
    def clean(self):
        self.writer.close()
        os.system(f"rm -rf logs/tensorboard-log/{self.mission}")


class model_judger(object):
    def __init__(self) -> None:
        self.best_score = -1
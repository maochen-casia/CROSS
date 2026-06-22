import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm

class Trainer:
    def __init__(self, 
                 config: OmegaConf,
                 model: Module, 
                 data_loader: DataLoader):
        
        self.model = model

        self.data_loader = data_loader
        self.num_epochs = config.num_epochs
        self.cur_epoch = 1
        train_params = self.model.train_params()
        self.train_params = [{'params': param['params'], 'lr': param['lr_scale'] * config.lr} for param in train_params]

        optimizer_class = getattr(torch.optim, config.optimizer)

        self.optimizer = optimizer_class(self.train_params,
                                         weight_decay=config.weight_decay,
                                         betas=config.betas,
                                         eps=config.eps)
        
        self.scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=config.num_warmup_steps,
                                                         num_training_steps=len(self.data_loader)*self.num_epochs)
        self.grad_clip_max_norm = config.grad_clip_max_norm
    
        self.mixed_precision = config.mixed_precision
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

    def state_dict(self):
        state_dict = {'model': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'scheduler': self.scheduler.state_dict()}
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def zero_grad(self):
        for param in self.train_params:
            for p in param['params']:
                p.grad = None

    def step(self, data):
        self.zero_grad()
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                pred = self.model(data['input'])
                loss, loss_dict = self.model.loss(pred, data['label'])
            self.scaler.scale(loss).backward()
            if self.grad_clip_max_norm is not None and self.grad_clip_max_norm > 0:
                # Unscale before clipping so max-norm is applied on true gradient values.
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            pred = self.model(data['input'])
            loss, loss_dict = self.model.loss(pred, data['label'])
            loss.backward()
            if self.grad_clip_max_norm is not None and self.grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
            self.optimizer.step()

        self.scheduler.step()
        return loss_dict
    
    def train(self):
        self.model.train()
        process = tqdm(self.data_loader, desc=f'[Epoch {self.cur_epoch}/{self.num_epochs}] Training')
        for data in process:
            loss_dict = self.step(data)
            process.set_postfix(loss_dict)
        process.close()
        self.cur_epoch += 1
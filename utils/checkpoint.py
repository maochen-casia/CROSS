import torch
import os
from omegaconf import OmegaConf
from copy import deepcopy

class Checkpoint:
    def __init__(self, config: OmegaConf):
        
        self.config = config
        self.start_epoch = 1
        self.best_val_loss = float('inf')
        self.best_val_param = None
        self.trainer_state = None

        exp_name = config.exp_name
        self.save_dir = save_dir = f'./checkpoints/{exp_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_path = os.path.join(save_dir, 'checkpoint.pth')

        # load from last checkpoint
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path, weights_only=False)
            self.config = checkpoint['config']
            self.start_epoch = checkpoint['start_epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.best_val_param = checkpoint['best_val_param']
            self.trainer_state = checkpoint['trainer_state']
            print(f'Checkpoint {exp_name} already exists, loading from last checkpoint.')
    
    def set_trainer(self, trainer):
        # assign trainer, and load trainer param
        self.trainer = trainer
        self.trainer.cur_epoch = self.start_epoch
        if self.trainer_state is not None:
            self.trainer.load_state_dict(self.trainer_state)
    
    def step(self, val_loss, save_current_weights=False):

        self.start_epoch += 1
        self.trainer_state = self.trainer.state_dict()

        # update best validation model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_param = deepcopy(self.trainer.model.state_dict())
            print('Best validation model has been updated.')
        
        # save checkpoint
        checkpoint = {'config':self.config,
                      'start_epoch':self.start_epoch,
                      'best_val_loss':self.best_val_loss,
                      'best_val_param':self.best_val_param,
                      'trainer_state':self.trainer_state}
        torch.save(checkpoint, self.save_path)

        if save_current_weights:
            current_save_path = os.path.join(self.save_dir, f'checkpoint_epoch_{self.start_epoch-1}.pth')
            torch.save(self.trainer.model.state_dict(), current_save_path)
from omegaconf import OmegaConf
from torch.nn import Module
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from .metrics import get_metric

class Evaluator:
    def __init__(self, 
                 config: OmegaConf,
                 model: Module, 
                 data_loader: DataLoader):
        
        self.model = model
        self.data_loader = data_loader
        self.metrics = {k: get_metric(metric_name=k) for k in config.metrics}
        self.loss_metric_name = config.loss_metric # used to save best validation model

    def step(self, data):
        pred = self.model.forward(data['input'])
        label = data['label']
        for k, metric in self.metrics.items():
            metric(pred, label)

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    @torch.no_grad()
    def evaluate(self):

        self.model.eval()
        self.reset_metrics()

        process = tqdm(self.data_loader, desc='Evaluating')
        for data in process:
            self.step(data)

        # aggregate metrics
        outputs = {}
        info = ''
        for k, metric in self.metrics.items():
            outputs[k] = metric.aggregate()
            info += f'{k}: {outputs[k]}, '
        outputs['loss'] = outputs[self.loss_metric_name]

        info = info[:-2]

        return outputs, info
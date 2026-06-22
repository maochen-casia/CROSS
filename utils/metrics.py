import torch
import numpy as np

from transform import matrix_to_euler_zyx

def get_metric(metric_name):    
    if metric_name == 'distance_mean':
        return DistanceMeanMetric()
    elif metric_name == 'distance_median':
        return DistanceQuantileMetric(0.5)
    elif metric_name == 'yaw_mean_deg':
        return YawMeanMetric(in_degrees=True)
    elif metric_name == 'yaw_median_deg':
        return YawMedianMetric(in_degrees=True)

def wrap_angle_rad(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))

def translation_error(pred, label):
    t_left2world_pred = pred['t_left2world']
    t_left2world_label = label['t_left2world'].to(t_left2world_pred.device)
    error = torch.norm(t_left2world_pred - t_left2world_label, dim=-1)
    return error


def yaw_error(pred, label, in_degrees: bool = False):
    R_left2world_pred = pred['R_left2world']
    R_left2world_label = label['R_left2world'].to(R_left2world_pred.device)
    yaw_pred = matrix_to_euler_zyx(R_left2world_pred)[2]
    yaw_label = matrix_to_euler_zyx(R_left2world_label)[2]
    error = torch.abs(wrap_angle_rad(yaw_pred - yaw_label))
    if in_degrees:
        error = error * (180.0 / np.pi)
    return error


class BaseMetric:

    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, pred, label):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def aggregate(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
class DistanceMeanMetric(BaseMetric):
    def __init__(self):
        super().__init__()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def __call__(self, pred, label):
        error = translation_error(pred, label)
        self.total += error.sum().item()
        self.count += error.shape[0]
        return error
    
    def aggregate(self):
        if self.count == 0:
            return 0
        return self.total / self.count


class DistanceQuantileMetric(BaseMetric):
    def __init__(self, quantile):
        super().__init__()
        self.quantile = quantile

    def reset(self):
        self.errors = torch.empty(0)
        self.count = 0

    def __call__(self, pred, label):
        error = translation_error(pred, label)
        self.errors = torch.cat([self.errors.to(error.device), error], dim=0)
        self.count += error.shape[0]
        return error
    
    def aggregate(self):
        if self.count == 0:
            return 0
        return torch.quantile(self.errors, self.quantile)

class YawMeanMetric(BaseMetric):
    def __init__(self, in_degrees: bool = False):
        self.in_degrees = in_degrees
        super().__init__()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def __call__(self, pred, label):
        error = yaw_error(pred, label, in_degrees=self.in_degrees)
        self.total += error.sum().item()
        self.count += error.shape[0]
        return error

    def aggregate(self):
        if self.count == 0:
            return 0
        return self.total / self.count


class YawMedianMetric(BaseMetric):
    def __init__(self, in_degrees: bool = False):
        self.in_degrees = in_degrees
        super().__init__()

    def reset(self):
        self.errors = torch.empty(0)
        self.count = 0

    def __call__(self, pred, label):
        error = yaw_error(pred, label, in_degrees=self.in_degrees)
        self.errors = torch.cat([self.errors.to(error.device), error], dim=0)
        self.count += error.shape[0]
        return error

    def aggregate(self):
        if self.count == 0:
            return 0
        return torch.quantile(self.errors, 0.5).item()

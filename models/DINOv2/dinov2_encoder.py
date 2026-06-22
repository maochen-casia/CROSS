import os
import sys

code_dir = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = 'your/path/to/DINOv2/repo'
if REPO_DIR not in sys.path:
    sys.path.append(REPO_DIR)

weight_paths = {
    'dinov2_vitl14': 'your/path/to/dinov2/pretrained/weight/dinov2_vitl14_pretrain.pth',
}

import torch
import torch.nn as nn


class DINOv2(nn.Module):

    def __init__(self, model_name, device, freeze=True):
        super().__init__()

        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

        assert model_name in weight_paths.keys(), f"Model name {model_name} not recognized. Available models: {list(weight_paths.keys())}"
        weight_path = weight_paths[model_name]
        self.pretrained = torch.hub.load(REPO_DIR, model_name, source='local', weights=weight_path)
        self.pretrained.eval()

        self.device = device
        self.to(device)

        self.freeze = freeze
        if self.freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.num_intermediate_layers = 4
        self.embed_dim = {
            'dinov2_vitl14': 1024,
        }[model_name]

        self.scale = 14

    def train(self, mode=True):
        if self.freeze:
            return
        super().train(mode)

    def pre_process(self, images):
        images = images.to(self.device)
        images = images.to(self.dtype)
        images = (images - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        return images

    def get_intermediate_layers(self, images, return_class_token=True, reshape=False):
        images = self.pre_process(images)
        if self.freeze:
            with torch.no_grad(), torch.autocast('cuda', dtype=self.dtype):
                features = self.pretrained.get_intermediate_layers(
                    images,
                    n=self.num_intermediate_layers,
                    return_class_token=return_class_token,
                    reshape=reshape,
                )
        else:
            features = self.pretrained.get_intermediate_layers(
                images,
                n=self.num_intermediate_layers,
                return_class_token=return_class_token,
                reshape=reshape,
            )
        features = [(f[0].float(), f[1].float()) for f in features]
        return features

    def forward(self, images, reshape=False):
        images = self.pre_process(images)

        if self.freeze:
            with torch.no_grad(), torch.autocast('cuda', dtype=self.dtype):
                features = self.pretrained.forward_features(images)['x_norm_patchtokens']
        else:
            features = self.pretrained.forward_features(images)['x_norm_patchtokens']

        if reshape:
            h, w = images.shape[2] // self.scale, images.shape[3] // self.scale
            features = features.permute(0, 2, 1).unflatten(2, (h, w))

        return features.float()

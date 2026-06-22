import os, sys

code_dir = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = 'your/path/to/DINOv3/repo'
if REPO_DIR not in sys.path:
    sys.path.append(REPO_DIR)

weight_paths = {
    'dinov3_vitl16': 'your/path/to/DINOv3/pretrained/weight/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
}

import torch
import torch.nn as nn

class DINOv3(nn.Module):

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
        #self.to(self.dtype)

        self.intermediate_layer_idx = {
            'dinov3_vitb16': [2, 5, 8, 11],
            'dinov3_vitl16': [4, 11, 17, 23]
        }[model_name]

        self.embed_dim = {
            'dinov3_vitb16': 768,
            'dinov3_vitl16': 1024
        }[model_name]

        self.scale = 16

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
                features =  self.pretrained.get_intermediate_layers(images, n=self.intermediate_layer_idx, 
                                                                    return_class_token=return_class_token,
                                                                    reshape=reshape)
        else:
            features = self.pretrained.get_intermediate_layers(images, n=self.intermediate_layer_idx, 
                                                               return_class_token=return_class_token,
                                                               reshape=reshape)
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
            features = features.permute(0,2,1).unflatten(2, (h, w))

        return features.float()

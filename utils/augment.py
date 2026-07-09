import albumentations as A
import numpy as np
import PIL.Image as Image

class Augmentation:
    def __init__(self, seed):
        self.aug = A.Compose([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                    A.OneOf([
                        A.GaussNoise(p=0.5),
                        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
                    ], p=0.8),
                    A.OneOf([
                        A.MotionBlur(blur_limit=7, p=0.5),
                        A.GaussianBlur(blur_limit=7, p=0.5),
                    ], p=0.6),
                    A.CoarseDropout(num_holes_range=(4, 8), 
                                    hole_height_range=(1/64, 1/4), 
                                    hole_width_range=(1/64, 1/4), 
                                    fill=0, 
                                    p=0.7),
                ], seed=seed)
    
    def __call__(self, image: Image.Image) -> Image.Image:
        image = np.array(image)
        augmented = self.aug(image=image)
        image = augmented['image']
        image = Image.fromarray(image)
        return image
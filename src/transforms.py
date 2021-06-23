import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_transformations(aug_name='soft', image_size=(400, 400)):
    all_transforms = {
        'soft' : A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Rotate(p=0.3, limit=(-15, 15), interpolation=0, border_mode=2),
            A.IAAPerspective (scale=(0.05, 0.1), keep_size=True, p=0.5),
            A.RandomBrightnessContrast(p=0.3, 
                                       brightness_limit=(-0.15, 0.15), 
                                       contrast_limit=(-0.15, 0.15)),
            A.Blur(p=0.3, 
                   blur_limit=(3, 7)),
            A.Cutout(p=0.5, 
                     num_holes=8, 
                     max_h_size=20, 
                     max_w_size=20),
            A.HueSaturationValue(p=0.3, 
                                 hue_shift_limit=(-20, 20), 
                                 sat_shift_limit=(-30, 30), 
                                 val_shift_limit=(-20, 20)),
            A.GaussNoise(var_limit=5. / 255., p=0.05),
            A.ISONoise(p=0.2, 
                       intensity=(0.1, 0.5), 
                       color_shift=(0.01, 0.05)),
            A.MotionBlur(p=0.1, 
                         blur_limit=(3, 7)),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='yx', 
                                            remove_invisible=False,
                                            label_fields=['line_idxs'])),
        'test_aug' : A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        'no_aug' : A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='yx', 
                                            remove_invisible=False,
                                            label_fields=['line_idxs']))
    }

    image_transforms = all_transforms[aug_name]
    return image_transforms
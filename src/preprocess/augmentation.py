import albumentations as A
import numpy as np


class ObjectDetectionAugmentation:
    def __init__(self, config: dict):
        '''
        Args:
            config: dictionaly of augmentation params
            (example)
            {
                rotation: {p: 0.5},
                vertical_flip: {p: 0.5},
                horizontal_flip: {p: 0.5},
                random_crop: {width: 450, height: 450},
                random_brightness_contrast: {p: 0.2},
                bbox_param: {format: 'coco', min_area: 100, min_visibility: 0.1, label_fields: []}
            }
        '''
        self.config = config
        self.transformer = A.Compose(
            [
                A.RandomCrop(**config['random_crop']),
                A.HorizontalFlip(**config['horizontal_flip']),
                A.VerticalFlip(**config['vertical_flip']),
                A.RandomBrightnessContrast(**config['random_brightness_contrast']),
            ],
            bbox_params=A.BboxParams(**config['bbox_param'])
        )
    
    def transform(self, img: np.array, bboxes: list):
        '''Transform set of image and bboxes'''
        transformed = self.transformer(image=img, bboxes=bboxes)
        transformed_img = transformed['image']
        transformed_bboxes = transformed['bboxes']
        return transformed_img, transformed_bboxes

    def transform_data(self, img_list: list, bboxes: list):
        '''Transform all data'''
        transformed_img_list = []
        transformed_bboxes_list = []
        for _img, _bboxes in zip(img_list, bboxes):
            transformed_img, transformed_bboxes = self.transform(_img, _bboxes)
            transformed_img_list.append(transformed_img)
            transformed_bboxes_list.append(transformed_bboxes)
        return transformed_img_list, transformed_bboxes_list
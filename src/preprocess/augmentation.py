import albumentations as A
import numpy as np


class ObjectDetectionAugmentation:
    def __init__(self, config: dict):
        '''
        Args:
            config: dictionaly of augmentation params
            (example)
            {
                augment_args:
                    vertical_flip: {p: 0.5}
                    horizontal_flip: {p: 0.5}
                    random_crop: {width: 450, height: 450}
                    random_brightness_contrast: {p: 0.2}
                bbox_param: {format: 'coco', min_area: 100, min_visibility: 0.1, label_fields: []}
                compile: 'Compose'
            }
        '''
        # add other augmentations, if you want
        aug_list = []
        for key in config['augment_args']:
            if key == 'random_crop':
                aug = A.RandomCrop(**config['augment_args'][key])
            elif key == 'horizontal_flip':
                aug = A.HorizontalFlip(**config['augment_args'][key])
            elif key == 'vertical_flip':
                aug = A.VerticalFlip(**config['augment_args'][key])
            elif key =='random_brightness_contrast':
                aug = A.RandomBrightnessContrast(**config['augment_args'][key])
            aug_list.append(aug)
        
        if config['compile'] == 'compose':
            compile_augs = A.Compose
        elif config['compile'] == 'oneof':
            compile_augs = A.OneOf
            
        self.transformer = compile_augs(
            aug_list,
            bbox_params=A.BboxParams(**config['bbox_param'])
        )
        self.config = config
        
    
    def transform(self, img: np.array, bboxes: list):
        '''Transform set of image and bboxes
        Args:
            img(np.array): input image
            bboxes(list): list of bounding boxes
        Return:
            transformed_img(np.array): output image
            transformed_bboxes(list): list of bounding boxes
        '''
        transformed = self.transformer(image=img, bboxes=bboxes)
        transformed_img = transformed['image']
        transformed_bboxes = transformed['bboxes']
        return transformed_img, transformed_bboxes
    
    
    # Todo: Delete loop
    def transform_data(self, img_list: list, bboxes_list: list):
        '''Transform all data
       Args:
            img_list(list[np.array]): list of input images
            bboxes_list(list[list]): list of bounding boxes list
        Return:
            transformed_img_list(list[np.array]): list ofoutput image
            transformed_bboxes_list(list[list]): list of bounding boxes list
        '''
        transformed_img_list = []
        transformed_bboxes_list = []
        for _img, _bboxes in zip(img_list, bboxes_list):
            transformed_img, transformed_bboxes = self.transform(_img, _bboxes)
            transformed_img_list.append(transformed_img)
            transformed_bboxes_list.append(transformed_bboxes)
        return transformed_img_list, transformed_bboxes_list

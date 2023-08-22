#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#
# SPDX-License-Identifier: MIT


import numpy as np

from . import transforms
def label_mapper_crosscity_13():
    mapper =  {7: 0,
            8: 1,
            11: 2,
            12: 255,
            13: 255,
            17: 255,
            19: 3,
            20: 4,
            21: 5,
            22: 255,
            23: 6,
            24: 7,
            25: 8,
            28: 10,
            32: 11,
            33: 12,
            # comment the below to 13 classes
            26: 13,
            27: 14,
            28: 15,
            31: 16,
            32: 17,
            33: 18,
            # uncomment the below to 13 classes
            27: 255,
            26: 9,
            31: 255,
            }
    arr = 255 * np.ones((255, ))
    for x in mapper:
        arr[x] = mapper[x]
    return arr

def label_mapper_cityscapes_19():
    ignore_label = 255
    mapper = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}        
    arr = 255 * np.ones((255, ))
    for x in mapper:
        arr[x] = mapper[x]
    return arr

# label_mapper = label_mapper_crosscity_13
label_mapper = label_mapper_cityscapes_19

def get_train_transforms(args, mine_id):
    label_to_id = label_mapper()

    train_src_transforms = [
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomScaleCrop(args.base_size, args.input_size, args.train_scale_src, [], 0.0),
        transforms.DefaultTransforms(),
        transforms.RemapLabels(label_to_id)
    ]

    train_tgt_transforms = [
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomScaleCrop(args.base_size, args.input_size,
                                       args.train_scale_tgt, [], []),
        transforms.DefaultTransforms(),
        # transforms.RemapLabels(label_to_id)
    ]
    train_src_transforms = transforms.Compose(train_src_transforms)
    train_tgt_transforms = transforms.Compose(train_tgt_transforms)
    return train_src_transforms, train_tgt_transforms


def get_test_transforms():
    label_to_id = label_mapper()
    tgt_transforms = [
        transforms.Resize((1024, 512)), 
        transforms.DefaultTransforms(),
        transforms.RemapLabels(label_to_id)
    ]
    return transforms.Compose(tgt_transforms)


def get_val_transforms(args):
    label_to_id = label_mapper()
    tgt_transforms = [
        transforms.Resize((1024, 512)),
        transforms.DefaultTransforms(),
        # transforms.RemapLabels(label_to_id)  ## Getting rid of this because NTHU has no labels for train set. 
    ]
    return transforms.Compose(tgt_transforms)

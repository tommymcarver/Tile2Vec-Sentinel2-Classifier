import os
import sys
import numpy as np

def crop_notcrop_train(n_tiles, old_labels):
    new_labels = []
    not_crop =[111, 112, 121, 122, 123, 124, 131, 141, 142, 143, 152, 176, 190, 87, 88, 92, 63, 64]
    for idx, label in enumerate(old_labels):
        if (idx < n_tiles):
            for bad in not_crop:
                if label == bad:
                    new_labels.append(0)
                    break
            if len(new_labels) == idx:
                new_labels.append(1)
        else:
            break
    return new_labels

def crop_notcrop_test(n_tiles, old_labels):
    new_labels = []
    not_crop =[111, 112, 121, 122, 123, 124, 131, 141, 142, 143, 152, 176, 190, 87, 88, 92, 63, 64]
    count = 0
    for idx, label in enumerate(old_labels):
        if (idx >= (len(old_labels)-n_tiles)):
            for bad in not_crop:
                if label == bad:
                    new_labels.append(0)
                    break
            if (len(new_labels) + len(old_labels) - n_tiles) == idx:
                new_labels.append(1)
    return new_labels

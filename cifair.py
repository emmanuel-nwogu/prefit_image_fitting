""" ciFAIR data loaders for PyTorch.

Version: 1.0

https://cvjena.github.io/cifair/
"""
import numpy as np
import torchvision
import random
import torch

from typing import Any, Callable, Optional, Tuple
from PIL import Image


class TriplesCiFAIR10(torchvision.datasets.CIFAR10):
    base_folder = 'ciFAIR-10'
    url = 'https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-10.zip'
    filename = 'ciFAIR-10.zip'
    tgz_md5 = 'ca08fd390f0839693d3fc45c4e49585f'
    test_list = [
        ['test_batch', '01290e6b622a1977a000eff13650aca2'],
    ]

    def __init__(self, determinism_seed, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False, num_triples=100):
        random.seed(determinism_seed)
        super(TriplesCiFAIR10, self).__init__(root=root, train=train, transform=transform,
                                              target_transform=target_transform,
                                              download=download)
        self.num_triples = num_triples

        # Repeat the following `num_triples` times.
        # Pick a random image of class C0, A.
        # Pick another random image of class C1, B where C1 != C0 and A != B. Last conditional is redundant but still :)
        # Pick another random image of class C0, C where C != A.

        a_image_indices = random.sample(list(range(len(self.targets))), self.num_triples)
        b_selected_image_indices = set()
        c_selected_image_indices = set()
        abc_image_triples = list()
        abc_target_triples = list()
        for a_image_index_in_targets in a_image_indices:
            a_image_class = self.targets[a_image_index_in_targets]
            other_classes_index_to_class = {class_index: class_label for class_index, class_label in
                                            enumerate(self.targets) if
                                            class_label != a_image_class and class_index not in b_selected_image_indices}
            b_image_index_in_targets = random.sample(other_classes_index_to_class.keys(), 1)[0]

            same_class_index_to_class = {class_index: class_label for class_index, class_label in
                                         enumerate(self.targets) if
                                         class_label == a_image_class and class_index != a_image_index_in_targets and
                                         class_index not in c_selected_image_indices}
            c_image_index_in_targets = random.sample(same_class_index_to_class.keys(), 1)[0]

            abc_image_triples.append([self.data[i] for i in
                                      (a_image_index_in_targets, b_image_index_in_targets, c_image_index_in_targets)])
            abc_target_triples.append([self.targets[i] for i in
                                       (a_image_index_in_targets, b_image_index_in_targets, c_image_index_in_targets)])

            # Add the selected targets, so they don't get selected again.
            b_selected_image_indices.add(b_image_index_in_targets)
            c_selected_image_indices.add(c_image_index_in_targets)

        del self.data, self.targets  # We don't need these anymore :)

        self.abc_image_triples = np.array(abc_image_triples)
        self.abc_target_triples = np.array(abc_target_triples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        imgs, targets = self.abc_image_triples[index], self.abc_target_triples[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        imgs = [Image.fromarray(img) for img in imgs]

        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs]

        if self.target_transform is not None:
            targets = [self.target_transform(target) for target in targets]

        imgs = [img.permute(1, 2, 0).view(-1, 3) for img in imgs]
        return imgs, targets

    def __len__(self) -> int:
        return self.num_triples


class CIFAIR10(torchvision.datasets.CIFAR10):
    base_folder = 'ciFAIR-10'
    url = 'https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-10.zip'
    filename = 'ciFAIR-10.zip'
    tgz_md5 = 'ca08fd390f0839693d3fc45c4e49585f'
    test_list = [
        ['test_batch', '01290e6b622a1977a000eff13650aca2'],
    ]


class CIFAIR100(torchvision.datasets.CIFAR100):
    base_folder = 'ciFAIR-100'
    url = 'https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-100.zip'
    filename = 'ciFAIR-100.zip'
    tgz_md5 = 'ddc236ab4b12eeb8b20b952614861a33'
    test_list = [
        ['test', '8130dae8d6fc6a436437f0ebdb801df1'],
    ]


if __name__ == '__main__':
    cifair10_save_folder = r"data/cifair10"
    cifair10_dataset = TriplesCiFAIR10(root=cifair10_save_folder, train=True, download=True)
    dbg = 0

import os
import json
import numpy as np
import torch
import skimage.metrics
import matplotlib.pyplot as plt

from enum import Enum
from tqdm.autonotebook import tqdm


# From https://github.com/vsitzmann/siren/blob/master/dataio.py
def get_mgrid(sidelen, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1."""
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


class Params:
    def __init__(self, learning_rate=1e-5, reduce_lr_patience=500, weight_decay=0, batch_size=64, in_features=2,
                 hidden_features=256, hidden_layers=3, out_features=3, fit_epochs=50, num_triples=10):
        self.reduce_lr_patience = reduce_lr_patience
        self.weight_decay = weight_decay
        self.num_triples = num_triples
        self.fit_epochs = fit_epochs
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def save_json(self, save_path="", filename="params.json"):
        save_path = os.path.abspath(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        parameters_dict = self.__dict__
        full_path = os.path.join(save_path, filename)
        try:
            with open(full_path, 'w') as param_json_file:
                json.dump(parameters_dict, param_json_file)
        except OSError:
            raise ValueError(f"Saving parameters to JSON file at {full_path} failed!")

    @classmethod
    def create_from_json(cls, json_full_path):
        try:
            with open(json_full_path, 'r') as param_json_file:
                new_obj = Params()
                valid_keys = list(new_obj.__dict__)
                parameters_dict = json.load(param_json_file)
                parameters_dict = {k: v for k, v in parameters_dict.items() if k in valid_keys}
                new_obj.__dict__.update(parameters_dict)
                return new_obj
        except Exception:
            print(f"Creating {cls.__name__} object from JSON file failed! File path: {json_full_path}.")
            raise


class FittingMode(Enum):
    FIT_FROM_RANDOM_INIT = 1
    FIT_FROM_MODEL_FIT_TO_IMAGE_FROM_SAME_CLASS = 2
    FIT_FROM_MODEL_FIT_TO_IMAGE_FROM_DIFFERENT_CLASS = 3

    def title_case_name(self):
        enum_to_title = {self.FIT_FROM_RANDOM_INIT: "Fit on image A from random initialisation",
                         self.FIT_FROM_MODEL_FIT_TO_IMAGE_FROM_SAME_CLASS:
                             "Fit on image A using model pre-fit on an image from the same class",
                         self.FIT_FROM_MODEL_FIT_TO_IMAGE_FROM_DIFFERENT_CLASS:
                             "Fit on image A using model pre-fit on an image from a different class"}
        return enum_to_title[self]


def calculate_psnr_for_one_image(pred_img, gt_img):
    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()
    pred_img, gt_img = [(image / 2.) + 0.5 for image in (pred_img, gt_img)]
    return skimage.metrics.peak_signal_noise_ratio(pred_img, gt_img, data_range=1)


def fit_one_image(model, coords, image_to_fit, params: Params, device, tqdm_description="", plot_losses=False,
                  current_fitting_mode: FittingMode = None, pretrain=True, calculate_psnr=False):
    assert image_to_fit.shape[0] == 1, "This function only supports a single image in a given batch!"
    image_to_fit = image_to_fit[0]
    optim = torch.optim.Adam(lr=params.learning_rate, params=model.parameters(), weight_decay=params.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5,
                                                           patience=params.reduce_lr_patience, threshold=0.0001,
                                                           threshold_mode='rel', cooldown=0, eps=1e-08, verbose=True)
    with tqdm(total=params.fit_epochs, desc=tqdm_description, leave=True) as pbar:
        metrics = {"mses": [], "psnrs": []}
        for epoch in range(params.fit_epochs):
            coords = coords.to(device)
            image_to_fit = image_to_fit.to(device)
            predicted_image = model(coords)
            loss = ((predicted_image - image_to_fit) ** 2).mean()
            train_loss = loss.item()
            metrics["mses"].append(train_loss)

            if calculate_psnr:
                current_psnr = calculate_psnr_for_one_image(predicted_image, image_to_fit)
                metrics["psnrs"].append(current_psnr)

            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.update(1)
            scheduler.step(train_loss)

            if plot_losses:
                plt.figure(current_fitting_mode.value)
                plt.plot(metrics["mses"], color="b")
                plt.xlabel("Step")
                plt.ylabel("Training loss")
                if pretrain:
                    title = f"Pretraining: {current_fitting_mode.name}"
                else:
                    title = f"Fitting: {current_fitting_mode.name}"
                plt.title(title)
                plt.show(block=False)
                plt.pause(0.00001)
    return metrics


def save_run_info(run_info_dict, save_path, clear_existing_file=False):
    save_path = os.path.abspath(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filepath = os.path.join(save_path, "run_info.txt")
    if not os.path.isfile(filepath):
        run_info_file = open(filepath, "w")
        run_info_file.close()

    with open(filepath, "a") as run_info_file:
        if clear_existing_file:
            run_info_file.truncate(0)
        run_info_file.write(json.dumps(run_info_dict))
        run_info_file.write("\n")


if __name__ == '__main__':
    lol = Params.create_from_json("params.json")
    fitting_mode = FittingMode(1)

    dbg = 0

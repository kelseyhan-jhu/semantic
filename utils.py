import os
from scipy import stats
import torch
from PIL import Image
from torchvision.transforms import functional as tr

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

def listdir(dir, path=True):
    files = os.listdir(dir)
    files = [f for f in files if (f != '.DS_Store' and f != '._.DS_Store' and f != '.ipynb_checkpoints')]
    files = sorted(files)
    if path:
        files = [os.path.join(dir, f) for f in files]
    return files

def p2r(p, n):
    t = stats.t.ppf(1-p, n-2);
    r = (t**2/((t**2)+(n-2))) ** 0.5;
    return r

def image_to_tensor(image, resolution=None, do_imagenet_norm=True):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    if resolution is not None:
        image = tr.resize(image, resolution)
    if image.width != image.height:     # if not square image, crop the long side's edges
        r = min(image.width, image.height)
        image = tr.center_crop(image, (r, r))
    image = tr.to_tensor(image)
    if do_imagenet_norm:
        image = imagenet_norm(image)
    return image

def imagenet_norm(image):
    dims = len(image.shape)
    if dims < 4:
        image = [image]
    image = [tr.normalize(img, mean=imagenet_mean, std=imagenet_std) for img in image]
    image = torch.stack(image, dim=0)
    if dims < 4:
        image = image.squeeze(0)
    return image

def imagenet_unnorm(image):
    mean = torch.tensor(imagenet_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(imagenet_std, dtype=torch.float32).view(3, 1, 1)
    image = image.cpu()
    image = image * std + mean
    return image


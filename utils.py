import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader

def add_noise(images):
    """
    Add noise to images
    Args:
        images (torch.Tensor): images
    Returns:
        noisy_images (torch.Tensor): noisy images
    """

    # Adding non destructive noise to images - structure of image remains the same but the pixel values are changed
    noisy_images = transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(images)
    # noisy_images = transforms.RandomGrayscale(p=0.2)(noisy_images)
    noisy_images = transforms.GaussianBlur(3, sigma=(0.1, 2.0))(noisy_images)
    return noisy_images

def softmax_mse_loss(input, target):
    """
    Compute the softmax cross entropy loss
    Args:
        input (torch.Tensor): input
        target (torch.Tensor): target
    Returns:
        torch.Tensor: softmax cross entropy loss
    """
    return torch.mean(torch.sum(F.softmax(input, dim=1) * F.mse_loss(input, target, reduction='none'), dim=1))

def update_ema(model, ema_model, alpha, global_step):
    """
    Update the ema model weights with the model weights
    Args:
        model (torch.nn.Module): model
        ema_model (torch.nn.Module): ema model
        alpha (float): alpha
        global_step (int): global step
    """
    
    # Set alpha to 0.999 at the beginning and then linearly decay
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def accuracy_fn(y_pred, y_true):
    """
    Accuracy Metric to Evaluate Model
    Args:
        y_pred (torch.Tensor): model output raw logits
        y_true (torch.Tensor): ground truth segmentation masks
    Returns:
        accuracy (float): accuracy of the model
    """

    # Convert raw model output to probabilities
    sig_fn = nn.Sigmoid()
    y_pred_prob = sig_fn(y_pred)
    # Threshold Probabilties to Produce Class Labels
    y_pred_label = (y_pred_prob > 0.5).float()
    # Compute number of equal elements
    correct = torch.eq(y_pred_label, y_true).sum().item()
    # Divide by Number of elements to compute accuracy
    acc = (correct/torch.numel(y_true)) * 100
    return(acc)



def dice_loss(ps, ts):
    return - dice_score(ps, ts)


def dice_binary(ps, ts):
    ps = torch.round(ps).to(ps.dtype)
    ts = torch.round(ts).to(ts.dtype)
    return dice_score(ps, ts)


def dice_score(ps, ts, eps=1e-7):
    numerator = torch.sum(ts * ps, dim=(1, 2, 3)) * 2 + eps
    denominator = torch.sum(ts, dim=(1, 2, 3)) + \
        torch.sum(ps, dim=(1, 2, 3)) + eps
    return numerator / denominator

def get_dataset_mean_std():
    """
    Compute the mean and standard deviation of the dataset for normalization
    For Pet Dataset:
        mean: [0.4814, 0.4494, 0.3958]
        std:  [0.2563, 0.2516, 0.2601]
    Returns:
        mean (torch.Tensor): mean of the dataset
        std (torch.Tensor): standard deviation of the dataset
    """

    transform = transforms.Compose([
                transforms.Resize(size = (64,64)),

                transforms.ToTensor(),
                transforms.Normalize(mean = (0, 0, 0),
                    std  = (1, 1, 1)),
            ])
    mask_transform = transforms.Compose([
                transforms.Resize(size = (64,64)),
                transforms.PILToTensor()
            ])

    dataset = OxfordIIITPet(
            root = "data",
            download = True,
            target_types = "segmentation",
            transform = transform,
            target_transform = mask_transform
            )

    dataset_test = OxfordIIITPet(
            root = "data",
            split = "test",
            download = True,
            target_types = "segmentation",
            transform = transform,
            target_transform = mask_transform
            )

    image_loader = DataLoader(dataset + dataset_test, 
                            batch_size  = 1, 
                            shuffle     = False, 
                            num_workers = 2,
                            pin_memory  = True)

    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    for i, (image, label) in enumerate(image_loader):
        psum    += image.sum(axis        = [0, 2, 3])
        psum_sq += (image ** 2).sum(axis = [0, 2, 3])

    # pixel count
    image_size = 64
    count = len(image_loader) * image_size * image_size

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)
    
    return total_mean, total_std

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision.transforms import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models.segmentation import deeplabv3_resnet50
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss,
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics import Dice
import pickle

from dataset import LabeledUnlabeledPetDataset
from data import TwoStreamBatchSampler, SingleStreamBaselineSampler
from utils import accuracy_fn, update_ema, softmax_mse_loss
from ramp_up import get_current_consistency_weight

# Check if the GPU is available
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {DEVICE}')

NO_LABEL = -1
CUDA_LAUNCH_BLOCKING=1
global_step = 0

def train(train_loader, model, ema_model, optimizer, epoch, batch_size, labeled_batch_size, alpha):
    """
    Train the model
    Args:
        train_loader (torch.utils.data.DataLoader): train loader
        model (torch.nn.Module): model
        ema_model (torch.nn.Module): ema model
        optimizer (torch.optim.Optimizer): optimizer
        epoch (int): epoch
        batch_size (int): batch size
        labeled_batch_size (int): labeled batch size
        alpha (float): alpha
    """

    global global_step

    # Initialize the loss functions
    class_criterion = SoftBCEWithLogitsLoss(ignore_index=NO_LABEL)
    consistency_criterion = softmax_mse_loss

    # Switch to train mode
    model.train()
    ema_model.train()

    # Initialize accuracy and loss 
    model_accuracy = 0
    ema_model_accuracy = 0
    train_loss = 0
    dice, ema_dice = 0, 0
    jaccard, ema_jaccard = 0, 0
    dice_score = Dice().to(DEVICE)
    jaccard_score = BinaryJaccardIndex().to(DEVICE)

    # Timer
    epoch_start = time.time()

    for i, (images, ema_images, labels, true_labels) in enumerate(train_loader):

        # Move the data to the GPU
        input_var = images.to(DEVICE)
        ema_input_var = ema_images.detach().to(DEVICE)
        labels = labels.to(DEVICE, non_blocking=True)
        true_labels = true_labels.detach().to(DEVICE)

        # Compute the model output
        ema_model_out = ema_model(ema_input_var)["out"]
        model_out = model(input_var)["out"]

        ema_logit = ema_model_out
        ema_logit = ema_logit.detach().clone().requires_grad_(False)
        
        # Compute the classification loss
        class_loss = class_criterion(model_out.squeeze(), labels.squeeze().float()) / batch_size
        ema_class_loss = class_criterion(ema_logit.squeeze(), labels.squeeze().float()) / batch_size
        
        # Compute the consistency loss
        consistency_weight = get_current_consistency_weight(epoch)
        consistency_loss = consistency_weight * consistency_criterion(model_out, ema_logit) / batch_size
        
        # Compute total loss
        loss = class_loss + consistency_loss
        train_loss += loss.item()

        # Compute the accuracy
        model_accuracy += accuracy_fn(model_out, true_labels)
        ema_model_accuracy += accuracy_fn(ema_model_out, true_labels)
        # Compute the dice score
        dice += dice_score(model_out.squeeze(), true_labels.squeeze()).item()
        ema_dice += dice_score(ema_model_out.squeeze(), true_labels.squeeze()).item()
        # Compute the jaccard score
        jaccard += jaccard_score(model_out, true_labels).item()
        ema_jaccard += jaccard_score(ema_model_out, true_labels).item()
        
        # Backward pass 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        # Update the EMA model parameters
        update_ema(model, ema_model, alpha, global_step)

    epoch_end = time.time()

    model_accuracy = model_accuracy / len(train_loader)
    ema_model_accuracy = ema_model_accuracy / len(train_loader)
    dice = dice / len(train_loader)
    ema_dice = ema_dice / len(train_loader)
    jaccard = jaccard / len(train_loader)
    ema_jaccard = ema_jaccard / len(train_loader)
    train_loss = train_loss / len(train_loader)

    # Save the training metrics for analysis
    train_student_loss.append(train_loss)
    train_teacher_loss.append(train_loss)
    train_student_acc.append(model_accuracy)
    train_teacher_acc.append(ema_model_accuracy)
    train_student_dice.append(dice)
    train_teacher_dice.append(ema_dice)
    train_student_jaccard.append(jaccard)
    train_teacher_jaccard.append(ema_jaccard)

    print(f"Epoch: {epoch + 1} | Training loss: {round(loss.item(), 3)} | Class Loss: {round(class_loss.item(), 3)} | EMA Class Loss: {round(ema_class_loss.item(), 3)} | Consistency Loss: {round(consistency_loss.item(), 3)} | Student Train Acc: {round(model_accuracy, 3)} | Teacher Train Acc: {round(ema_model_accuracy, 3)} | Student Train Dice: {round(dice, 3)} | Teacher Train Dice: {round(ema_dice, 3)} | Student Train Jaccard: {round(jaccard, 3)} | Teacher Train Jaccard: {round(ema_jaccard, 3)} | Time: {round(epoch_end - epoch_start, 3)} ")

def eval(test_loader, model, ema_model, epoch):
    """
    Evaluate the model on the test set
    Args:
        test_loader (DataLoader): PyTorch dataloader for the test set
        model (nn.Module): The model to evaluate
        ema_model (nn.Module): The EMA model to evaluate
        epoch (int): The current epoch
    """
    model.eval()
    ema_model.eval()

    test_loss, ema_test_loss = 0, 0
    model_accuracy = 0
    ema_model_accuracy = 0
    dice, ema_dice = 0, 0
    jaccard, ema_jaccard = 0, 0
    dice_score = Dice().to(DEVICE)
    jaccard_score = BinaryJaccardIndex().to(DEVICE)

    class_criterion = nn.BCEWithLogitsLoss()

    with torch.inference_mode():

        for i, (images, labels) in enumerate(test_loader):
            # Move the images and labels to the GPU
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass of teacher and student models
            ema_model_out = ema_model(images)["out"]
            model_out = model(images)["out"]

            # Compute the loss
            loss = class_criterion(model_out.squeeze(), labels.squeeze().float())
            test_loss += loss.item()
            ema_loss = class_criterion(ema_model_out.squeeze(), labels.squeeze().float())
            ema_test_loss += ema_loss.item()

            # Compute the accuracy
            model_accuracy += accuracy_fn(model_out, labels)
            ema_model_accuracy += accuracy_fn(ema_model_out, labels)
            # Compute the dice score
            dice += dice_score(model_out.squeeze(), labels.squeeze()).item()
            ema_dice += dice_score(ema_model_out.squeeze(), labels.squeeze()).item()
            # Compute the jaccard score
            jaccard += jaccard_score(model_out, labels).item()
            ema_jaccard += jaccard_score(ema_model_out, labels).item()

    model_accuracy = model_accuracy / len(test_loader)
    ema_model_accuracy = ema_model_accuracy / len(test_loader)
    test_loss = test_loss / len(test_loader)
    dice = dice / len(test_loader)
    ema_dice = ema_dice / len(test_loader)
    jaccard = jaccard / len(test_loader)
    ema_jaccard = ema_jaccard / len(test_loader)

    # Save the test metrics for analysis
    test_student_loss.append(test_loss)
    test_teacher_loss.append(ema_test_loss)
    test_student_acc.append(model_accuracy)
    test_teacher_acc.append(ema_model_accuracy)
    test_student_dice.append(dice)
    test_teacher_dice.append(ema_dice)
    test_student_jaccard.append(jaccard)
    test_teacher_jaccard.append(ema_jaccard)

    spacer = " " * (len("Epoch:  | " + str(epoch)))
    print(f"{spacer}Test loss: {round(test_loss, 3)} | Student Test Acc: {round(model_accuracy, 3)} | Teacher Test Acc: {round(ema_model_accuracy, 3)} | Student Test Dice: {round(dice, 3)} | Teacher Test Dice: {round(ema_dice, 3)} | Student Test Jaccard: {round(jaccard, 3)} | Teacher Test Jaccard: {round(ema_jaccard, 3)}")

def main():
    global global_step

    print("==> Preparing data...")

    # Initialize the data loading parameters
    data_dir = 'data/oxford-iiit-pet'
    batch_size = 64
    labeled_batch_size = 32
    workers = 4
    unlabeled_ratio = 0.95

    # Download the dataset
    dataset = OxfordIIITPet(
        root = "data",
        download = True,
        target_types = "segmentation",
        )

    # Load the training data from custom dataset
    train_data = LabeledUnlabeledPetDataset(data_dir, train=True, labeled=False, unlabeled_ratio=unlabeled_ratio)
    # Create the batch sampler
    batch_sampler = TwoStreamBatchSampler(
                train_data.unlabeled_idxs, train_data.labeled_idxs, batch_size, labeled_batch_size)
    # Create the data loader
    train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_sampler=batch_sampler,
                                                num_workers=workers,
                                                pin_memory=True)

    test_data = LabeledUnlabeledPetDataset(data_dir, train=False, labeled=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=workers * 2)

    print("==> Building model...")

    # Initialize the model
    model = deeplabv3_resnet50(num_classes=1)
    model.to(DEVICE)
    model_ema = deeplabv3_resnet50(num_classes=1)
    model_ema.to(DEVICE)

    # Detach the EMA model parameters from the graph to prevent backprop
    for param in model_ema.parameters():
        param.detach_() 

    # Initialize the optimizer
    learning_rate = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the consistency loss
    alpha = 0.99

    print("==> Training model...")

    for epoch in range(25):
        train(train_loader, model, model_ema, optimizer, epoch, batch_size, labeled_batch_size, alpha)
        eval(test_loader, model, model_ema, epoch)

    print("==> Saving models...")
    torch.save(model.state_dict(), "model.pth")
    torch.save(model_ema.state_dict(), "model_ema.pth")

    # Save the training and test metrics for analysis
    with open('mean_teacher_0005.pkl', 'wb') as f: 
        pickle.dump([train_student_loss, train_teacher_loss, train_student_acc, train_teacher_acc, train_student_dice, train_teacher_dice, train_student_jaccard, train_teacher_jaccard, test_student_loss, test_teacher_loss, test_student_acc, test_teacher_acc, test_student_dice, test_teacher_dice, test_student_jaccard, test_teacher_jaccard], f)


if __name__ == "__main__":

    # Initialize the lists to save the training and test metrics
    train_student_loss, train_teacher_loss, train_student_acc, train_teacher_acc, train_student_dice, train_teacher_dice, train_student_jaccard, train_teacher_jaccard = [], [], [], [], [], [], [], []
    test_student_loss, test_teacher_loss, test_student_acc, test_teacher_acc, test_student_dice, test_teacher_dice, test_student_jaccard, test_teacher_jaccard = [], [], [], [], [], [], [], []

    main()
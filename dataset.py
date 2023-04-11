import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage import io
from PIL import Image
import random 

from data import TransformTwice

class LabeledUnlabeledPetDataset(Dataset):

    def __init__(self, data_dir, train, labeled=False, unlabeled_ratio=0.9, labeled_ratio=None):
        """
        Initialize the dataset.
        Args:
            data_dir (str): path to the dataset
            train (bool): train or test
            labeled (bool): labeled or unlabeled
            unlabeled_ratio (float): ratio of unlabeled data
            labeled_ratio (float): ratio of labeled data
        """
        self.img_labels = []
        self.img_dir = os.path.join(data_dir, "images")
        self.labels_dir = os.path.join(data_dir, "annotations")
        self.mask_dir = os.path.join(data_dir, "annotations", "trimaps")
        self.transform = transforms.Compose([
            transforms.Resize(size = (64,64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4814, 0.4494, 0.3958], 
                std=[0.2563, 0.2516, 0.2601])
        ])
        self.noise_transform = transforms.Compose([
            transforms.Resize(size = (64,64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0))
        ])
        self.train_transform = TransformTwice(self.transform, self.noise_transform)
        self.mask_transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize(size = (64,64))
        ])

        self.train = train
        self.train_file = 'trainval.txt'.format(self.labels_dir)
        self.test_file = 'test.txt'.format(self.labels_dir)
        self.labeled_idxs, self.unlabeled_idxs = [], []
        self.unlabeled_ratio = unlabeled_ratio

        # Load the data file
        self.fill_imgs(train=train, labeled=labeled)
        

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset given an index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = os.path.join(self.img_dir, self.img_labels[idx] + ".jpg")
        image = Image.open(img_path).convert('RGB')

        if self.train:
            image_ema = self.noise_transform(image)
            image = self.transform(image)
   
            seg_mask_path = os.path.join(self.mask_dir, self.img_labels[idx] + ".png")
            seg_mask = Image.open(seg_mask_path)
            seg_mask = self.mask_transform(seg_mask)
            seg_mask = torch.where(seg_mask > 1, 0, 1)
            
            # If labeled load trimap mask
            if (idx in self.labeled_idxs):
                return image, image_ema, seg_mask, seg_mask
            else:
                empty_mask = torch.full((1, 64, 64), -1, dtype=torch.int64)

                return image, image_ema, empty_mask, seg_mask
        else:
            image_ema = self.noise_transform(image)
            image = self.transform(image)

            seg_mask = seg_mask_path = os.path.join(self.mask_dir, self.img_labels[idx] + ".png")
            seg_mask = Image.open(seg_mask_path)
            seg_mask = self.mask_transform(seg_mask)
            seg_mask = torch.where(seg_mask > 1, 0, 1)

            return image, seg_mask
    
    def __getitemname__(self, idx):
        return self.img_labels[idx]
    
    def __getlabel__(self, idx):
        return idx in self.labeled_idxs
        
    def set_labels(self, prev_idx, file):
        """
        Set labels for the dataset.
        Args:
            prev_idx (int): List of tuples containing image names and labels.
        """
        num_files = len(file)
        num_unlabeled = int(num_files * self.unlabeled_ratio)
        
        # get random indices for labeled data 
        breed_labeled_idxs = random.sample(range(num_files), num_files - num_unlabeled)
        # add previous index to get absolute index 
        breed_labeled_idxs = [i + prev_idx for i in breed_labeled_idxs]
        # get unlabeled indices
        breed_unlabeled_idxs = [i for i in range(prev_idx, prev_idx + num_files) 
                                if i not in breed_labeled_idxs]

        return breed_labeled_idxs, breed_unlabeled_idxs
    
    def fill_imgs(self, train, labeled=False):
        """
        Fill the img_labels array and get indices.
        Args:
            labeled (bool): whether dataset should be labeled or partially unlabeled.
        """
        # Get the file
        file = self.train_file if train else self.test_file

        # Open the file
        with open(file) as lines:
            lines = lines.readlines()
            # sort by breed
            lines.sort()
            breed = None
            breed_files = []
            previous_index = 0

            if train:
                # Iterate through lines - each line is an image
                for i, line in enumerate(lines):
                    # Get the image filename and breed
                    filename = line.split(' ')[0]
                    line = filename.rsplit('_',1)[0]

                    # Fill array with image filename
                    self.img_labels.append(filename)

                    # If creating unlabeled dataset
                    if not labeled:
                        # If new breed
                        if line != breed:
                            
                            if breed is not None:
                                # Get labeled and unlabeled indices for breed
                                breed_labeled_idxs, breed_unlabeled_idxs = self.set_labels(previous_index, breed_files)
                                self.labeled_idxs += breed_labeled_idxs
                                self.unlabeled_idxs += breed_unlabeled_idxs

                            # Set new breed and set new breed list
                            breed = line
                            breed_files.clear()
                            breed_files.append(filename)
                            
                            # Set previous index to current index 
                            previous_index = i

                        else:
                            # Add to current file to list
                            breed_files.append(filename)

                # Get labeled and unlabeled indices for last breed
                breed_labeled_idxs, breed_unlabeled_idxs = self.set_labels(previous_index, breed_files)
                self.labeled_idxs += breed_labeled_idxs
                self.unlabeled_idxs += breed_unlabeled_idxs

                # Sort indices
                self.labeled_idxs.sort()
                self.unlabeled_idxs.sort()

            else:
                for i, line in enumerate(lines):
                    # Get the image filename and breed
                    filename = line.split(' ')[0]
                    line = filename.rsplit('_',1)[0]

                    # Fill array with image filename
                    self.img_labels.append(filename)

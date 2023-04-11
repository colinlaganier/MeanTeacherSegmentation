# Mean Teacher Image Segmentation

This is an implementation developed for the semi-supervised semantic segmentation task of the Oxford IIIT Pet dataset. This implementation is based on the work of The Curious AI Company and their publication: Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results ([1703.01780](https://arxiv.org/abs/1703.01780)).

TODO:
- Increase image size to 256px - 64px not enough but not enought GPU memory (AWS?)

## Package requirements:

- Pytorch
- TorchVision
- Numpy
- TorchMetrics
- [segmentation-models-pytorch](https://github.com/chsasank/segmentation_models.pytorch)

## Files:

- `main.py`: main script running creating the dataset, student and teacher models, optimizer, and running the training and evaluation. Saves a pickle file of the training and evaluation metrics as well as the two models.
- `dataset.py`: Custom `LabeledUnlabeledPetDataset` class to store sample and its associated labels, apply transforms as well as creating a the labelled unlabelled split of our data.
- `data.py`: Data loading utility functions, including the two stream sampler for the dataloader.
- `ramp_up.py`: Functions associated with the ramping up of the weights during training.
- `utils.py`: Utility functions.

To run the training and eval simply run:

```console
foo@bar:~/MeanTeacher$ python main.py
```

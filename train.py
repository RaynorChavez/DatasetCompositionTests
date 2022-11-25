import pytorch_lightning as pl
import os

import torch.utils.data.dataloader as DataLoader

from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.datamodules import ImagenetDataModule

from pl_bolts.models.self_supervised.simclr \
    import SimCLRTrainDataTransform
from pl_bolts.models.self_supervised.simclr \
    import SimCLREvalDataTransform


# Pytorch and Torchvision Imports
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils


if __name__=='__main__':
    if torch.cuda.is_available():
        # Use deterministic cudnn algorithms
        torch.backends.cudnn.deterministic = True
        epochs = 50
    else:
        device = torch.device("cpu")
        epochs = 5

    # os.environ["CUDA_VISIBLE_DEVICES"]="3"

    data_dir = './data/'
    BATCH_SIZE=32

    with torch.cuda.device("cuda:0"):
        dataset = dsets.ImageFolder(
            root=data_dir,
            transform=SimCLRTrainDataTransform(input_height=512)
        )
        dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)


        model = SimCLR(gpus=1, num_samples=1000, batch_size=BATCH_SIZE, dataset=dataset, max_epochs=epochs)

        print("Device: {}".format(torch.cuda.current_device()))
        print("Epochs: {}".format(epochs))

        # model = SimCLR(2, 5, 5, dataset, max_epochs=epochs)
        trainer = pl.Trainer(accelerator="gpu", max_epochs=epochs, default_root_dir="./trained_models/")
        trainer.fit(model, train_dataloaders=dataloader)


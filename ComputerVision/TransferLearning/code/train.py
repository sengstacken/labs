import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch.nn.functional as F
from torchvision import models
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
import torchmetrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

# MODEL FUNCTION REQUIRED FOR INFERENCE
def model_fn(model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RESNET()

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

class RESNET(pl.LightningModule):
    def __init__(self, batch_size=32, num_workers=4, num_classes=10):
        '''Constructor method 

        Parameters:
        train_data_dir (string): path of training dataset to be used either for training and validation
        batch_size (int): number of images per batch. Defaults to 128.
        test_data_dir (string): path of testing dataset to be used after training. Optional.
        num_workers (int): number of processes used by data loader. Defaults to 4.

        '''

        # Invoke constructor
        super(RESNET, self).__init__()

        # Set up class attributes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        
        # init a pretrained resnet
        backbone = models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters, self.num_classes)        

    def forward(self, x):
        '''Forward pass, it is equal to PyTorch forward method. Here network computational graph is built.  This is used for inference only and is seperate from the full training_step  

        Parameters:
        x (Tensor): A Tensor containing the input batch of the network

        Returns: 
        An one dimensional Tensor with probability array for each input image
        '''
        
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        #x = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        
        return x

    def configure_optimizers(self):
        '''
        Method to configure optmizers and learning rate schedulers
        
        Returns:
        (Optimizer): Adam optimizer tuned with model parameters
        '''
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        '''
        Method for full training loop.  Called for every training step, computes training loss, then logs and sends back 
        logs parameter to Trainer to perform backpropagation
        '''

        # Get input and output from batch
        x, labels = batch

        # Compute prediction through the network (forward pass)
        prediction = self.forward(x)

        # Compute training loss
        loss = F.cross_entropy(prediction, labels)
        
        self.train_acc(prediction, labels)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)

        # Logs training loss
        logs = {'train_loss': loss.detach()}

        output = {
            # This is required in training to be used by backpropagation
            'loss': loss,
            # This is optional for logging pourposes
            'log': logs
        }

        return output
    
    def validation_step(self, batch, batch_idx):
        '''
        Method for full training loop.  Called for every training step, computes training loss, then logs and sends back 
        logs parameter to Trainer to perform backpropagation
        '''
        # Get input and output from batch
        x, labels = batch

        # Compute prediction through the network (forward pass)
        prediction = self.forward(x)

        # Compute training loss
        loss = F.cross_entropy(prediction, labels)
        
        self.valid_acc(prediction, labels)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True, prog_bar=True)

        return {"val_loss": loss}
    
    def test_step(self, batch, batch_idx):
        x, labels = batch
        prediction = self.forward(x)
        loss = F.cross_entropy(prediction, labels)
        
        self.test_acc(prediction, labels)
        self.log('test_acc', self.test_acc)
        self.log('test_loss', loss)
        
        return {"test_loss": loss}
    
    def validation_epoch_end(self, outputs):
        # outputs = list of dictionaries
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        # use key 'log'
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS']) # used to support multi-GPU or CPU training
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    args, _ = parser.parse_known_args()
    print(args)
    
    pl.seed_everything(1234)
    
    ##########
    #  DATA  #
    ##########
    
    # transforms
    train_transforms = T.Compose([
            #T.RandomResizedCrop(224),
            T.Resize((224,224)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    val_transforms = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])      
   
    # datasets
    train_ds = ImageFolder(args.train, transform=train_transforms)
    val_ds = ImageFolder(args.validation, transform=val_transforms)
        
    print('TRAINING DATASET:')    
    print(train_ds)
    print('VALIDATION DATASET:')
    print(val_ds)
        
    # dataloaders
    train_loader = DataLoader(dataset=train_ds, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(dataset=val_ds, batch_size=args.batch_size, num_workers=4, shuffle=False)
      
    ############
    # TRAINING #
    ############
    
    # init trainer
    trainer = pl.Trainer(gpus=args.gpu_count, max_epochs=args.epochs)

    # init model
    model = RESNET( 
        batch_size=args.batch_size, 
        num_workers=4, 
        num_classes=10, 
        ) 

    print('Starting Model Training')
    trainer.fit(model, train_loader, val_loader)
    
    ############
    # TESTING  #
    ############

    result = trainer.test(dataloaders=val_loader)
    print(result)
    
    ############
    #  SAVING  #
    ############
    
    # After model has been trained, save its state into model_dir which is then copied to back S3
    print('Saving Trained Model')
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
    

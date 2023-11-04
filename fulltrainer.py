import inspect
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn import metrics
from IPython.display import clear_output


                                                   

import torch
from torchvision import models
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import pandas as pd 
import numpy as np
!pip install patchify --quiet
from patchify import patchify
import gc
import math


                    

from torch import nn
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} as a device')            

from tqdm.auto import tqdm
from typing import Dict, List, Tuple






# from torchvision import models
# models_dict = {'ResNet50': models.resnet50(weights = models.ResNet50_Weights.DEFAULT), 
#                'VGG16' : models.vgg16(weights = models.VGG16_Weights.DEFAULT), 
#                'DenseNet121' : models.densenet121(weights = models.DenseNet121_Weights.DEFAULT),
#                'EfficientNet_B0' : models.efficientnet_b0(weights = models.EfficientNet_B0_Weights.DEFAULT)}

# # model.eval() model is either of the models_dict



# # Step 4: Use the model and print the predicted category
# prediction = model(batch).squeeze(0).softmax(0)
# class_id = prediction.argmax().item()
# score = prediction[class_id].item()
# category_name = weights.meta["categories"][class_id]
# print(f"{category_name}: {100 * score:.1f}%")


# # model.train()


class HyperParameters:
    def __init__(self):
        self.hparams=None
    def save_hyperparameters(self,ignore=[]):
        '''saves hyperparameters in self.hparams'''
        frame=inspect.currentframe().f_back
        _,_,_,local_vars= inspect.getargvalues(frame)
        self.hparams={k:v for k,v in local_vars.items()
                     if k not in set(ignore+['self']) and not k.startswith('_')}
        
        for k,v in self.hparams.items():
            setattr(self,k,v)
    
class Utilities(HyperParameters):
    def __init__(self,y_label,y_pred):
        super().__init__()
        self.y_label=y_label
        self.y_pred=y_pred
    def _formaty(self,y):
        #y_pred=torch.reshape(self.y_pred,(-1,self.y_pred.shape[-1]))#mixes batches 
        if len(y.shape)!=1: #one hot encoded 
            y= torch.argmax(y,dim=1)
        return y
    def accuracy_fn(self):
        '''
        format should be for y_label no_of_result*onehot
        and for y_pred it is converted to total_sum*onehot
        '''
       
        #y_pred=torch.reshape(y_pred,(-1,y_pred.shape[-1]))#mixes batches 
        self.y_pred=self._formaty(self.y_pred)
        self.y_label=self._formaty(self.y_label)
        print(self.y_pred)
        print(self.y_label)
        correct_pred=torch.sum(self.y_pred==self.y_label).item()
        total= self.y_label.size(0)
        accuracy= correct_pred/total
        return accuracy
    
    def confusion_matrics(self):
        '''plots a confusion matrics and return confusion matrics'''
        self.y_pred=self._formaty(self.y_pred)
        self.y_label=self._formaty(self.y_label)
        display_labels=None #yet to code
        cm=confusion_matrix(self.y_label, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()
        return cm
    
    def f_score(self):
        '''returns f1 score and f2 score'''
        self.y_pred=self._formaty(self.y_pred)
        self.y_label=self._formaty(self.y_label)
        f1_score=metrics.f1_score(self.y_label,self.y_pred,average='macro')
        f2_score=metrics.fbeta_score(self.y_label,self.y_pred,beta=2,average='macro')
        return f1_score,f2_score
        
    def balanced_accuracy(self,sample_weight=None,adjusted=False):
        self.y_pred=self._formaty(self.y_pred)
        self.y_label=self._formaty(self.y_label)
        score_result =metrics.balanced_accuracy_score(self.y_label,self.y_pred,sample_weight=sample_weight,adjusted=adjusted)
        return score_result
    
    
class Per_Epoch(Utilities):
    def __init__(self,epoch,y_pred,y_label,checkpoint_path=None,log=True):
        super().__init__(y_pred,y_label)
        self.checkpoint_path=checkpoint_path
        self.epoch=epoch
        #self.model= model
        #self.optimizer=optimizer
        #self.loss=loss
        self.y_label=y_label
        self.y_pred=y_pred
        
    
        
    def plotperepoch(self,accuracy=None):
         # Example values for epochs 1 to 5
        if accuracy is None:
            raise ValueError( "required accuracy array but got None") 
        # Number of epochs (assuming accuracy_values represent epochs from 1 to N)
        clear_output(wait=True)
        epochs = len(accuracy_values)

        # Plotting accuracy vs epoch
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, epochs + 1), accuracy_values, marker='o', color='b', label='Accuracy')
        plt.title('Accuracy vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.xticks(range(1, epochs + 1))  # Optional: Set x-axis ticks to match epochs
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def checkpoints(self,model, optimizer,loss):
        if self.checkpoint_path is None:
            raise ValueError("Checkpoint path is not provided")
        
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
        }, checkpoint_path)
        
    def log(self,logfilepath):
        '''
        logs all utilities funtion 
        '''
        epoch_data = {
        "Epoch": self.epoch,
        "Accuracy": self.accuracy_fn(),
        "Confusion Matrix": self.confusion_matrics().tolist(),
        "F-Score": self.f_score(),
        "Balanced Accuracy": self.balanced_accuracy(),
        "HyperParameters":self.hparams if self.hparams is not None else "nan",
           }
        print(epoch_data)                                          
        if logfilepath is None:
            raise ValueError("log path is not provided")
        
        if not os.path.exists(logfilepath):
            os.makedirs(logfilepath)
                                                   
        json_file_path = "results2.json"
        mode = "a" if os.path.exists(json_file_path) else "w"
        with open(json_file_path, mode) as json_file:
            json.dump(epoch_data, json_file, indent=4)
            json_file.write('\n')  # Add a newline to separate JSON objects


class CancerDataSet(Dataset):
    '''
    write Image_path as column name
    '''
    def __init__(self,transform):
        
        self.path=None
        self.label=2
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL Image to float32 tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Applies normalization
        ])


        self.patches=np.array([])
        
    
    def __len__(self):
        return self.patches.shape[0]
    def __getitem__(self,index):
        img=self.patches[index]
        #label=self.label[index]# if images have different labels
        label=self.label
        if self.transform:
            img=self.transform(img)
        return (img,label)  
    def patch_creater(self,path,label,patchsize=256):
        
        self.path=path
        self.label=label
        print(self.path,self.label)
        if self.label is None :
            raise ValueError("label is None")
        img = Image.open(self.path)

        self.patches=np.array([])
        width, height = img.size
        subcrop=16
        subtiles=round(math.sqrt(subcrop))
        tile_size=(width//subtiles,height//subtiles)
        patchsize=256
        for i in range(subtiles):
                for j in range(subtiles):
                    left = tile_size[0] * i
                    upper = tile_size[1] * j
                    right = min(tile_size[0] * i + tile_size[0], width)
                    lower = min(tile_size[1] * j + tile_size[1], height)
                    print(left,upper,right,lower)
                    cropped_img = img.crop((left, upper, right, lower))
                    image = np.asarray(cropped_img)
        #             del img
                    image_height, image_width, channel_count = image.shape
                    patch_height, patch_width, step = patchsize, patchsize, patchsize
                    patch_shape = (patch_height, patch_width, channel_count)
                    patchedset = patchify(image, patch_shape, step=step)
                    del image
                    gc.collect()
                    patchedset=patchedset.reshape( -1,256, 256, 3)
                    if len(self.patches.shape) == 1:
                        self.patches = patchedset
                    else:
                        self.patches = np.concatenate((self.patches, patchedset), axis=0)

                    del patchedset
        return self.patches




class Models(Per_Epoch):
    def __init__(self):
        pass
        #super().__init__()
        
    def train_step(self,model,dataloader,device='cpu'):
        #                loss_fn: torch.nn.Module, 
#                optimizer: torch.optim.Optimizer,
        """Trains a PyTorch model for a single epoch.

        Turns a target PyTorch model to training mode and then
        runs through all of the required training steps (forward
        pass, loss calculation, optimizer step).

        Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

        Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
        """
        # Put model in train mode
        model.train()

        # Setup train loss and train accuracy values
        train_loss, train_acc = 0, 0
        print("yay i am inside train_step")
        # Loop through data loader data batches
        for batch, (X, y) in enumerate(self.train_loader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate  and accumulate loss
            loss = self.criterion(y_pred, y)
            train_loss += loss.item() 

            # 3. Optimizer zero grad
            self.optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            self.optimizer.step()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        # Adjust metrics to get average loss and accuracy per batch 
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        return train_loss, train_acc
    
    def test_step(self,model, 
              testdata,label,
              device='cpu'):
        model.eval()
        
        self.y_pred=model(testdata)
        self.y_label=label
        accuracy=self.accuracy_fn()
        cm=self.confusion_matrics()
        f1_score,f2_score=self.f_score()
        balanced_accuracy=self.balanced_accuracy()
        print(f"Accuracy: {accuracy}, Confusion Matrix: {cm}, F1 Score: {f1_score}, F2 Score: {f2_score}, Balanced Accuracy: {balanced_accuracy}")
        clear_output(wait=True)
        

from torchvision import transforms
class Trainer(CancerDataSet,Models):
    def __init__(self,transform, patch_size=256):
        super().__init__(transform=transform)
        self.cancer_data_set=CancerDataSet(transform)
        self.patch_size=patch_size
        self.optimizer=torch.optim.Adam(model.parameters())
        self.criterion=torch.nn.CrossEntropyLoss()
        self.train_loader=None
        self.dataset=None
        #all hyperparameters
        pass
    def training(self,df, model):
        for index,row in df.iterrows():
            path=row['Image_path']
            label=row['label']
            print(path, label,type(label))
            
            self.dataset=self.cancer_data_set.patch_creater(path,label,self.patch_size)
            self.patches=self.cancer_data_set.patches
            self.label=label
            #             print(len(dataset),"Dataset length")
            self.train_loader = DataLoader(self.cancer_data_set, batch_size=32, shuffle=True)
#             self.train_step(model,train_loader, loss_fn=criterion, optimizer=optimizer, device=device)
            self.train_step(model, self.train_loader)

#             self.test_step(models.resnet50)
        
#         return 
        #trainmodels 



from torch import nn
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} as a device')



from torchvision import transforms
patched_df=train=pd.read_csv('/kaggle/input/cancerdatasetwithpath/cancerdf.csv')
new_patched_df = patched_df[patched_df["image_id"] == 1080]
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                                    std = [0.229, 0.224, 0.225])])


model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
optimizer=torch.optim.Adam(model.parameters())
criterion=torch.nn.CrossEntropyLoss()
trainer = Trainer(transform = transform)
model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
trainer.training( new_patched_df,model)
# print(len(trainer))


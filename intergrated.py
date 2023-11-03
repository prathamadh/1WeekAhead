from torch.utils.data import Dataset,DataLoader
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import pandas as pd 
import numpy as np
from patchify import patchify
import gc
!pip install patchify --quiet
class CancerDataSet(Dataset):
    '''
    write Image_path as column name
    '''
    def __init__(self,transforms=None):
        
        self.path=None
        self.label=None
        self.transforms=transforms
        self.patchedset=np.array([])
        
    
    def __len__(self):
        return self.patchedset.shape[0]
    def __getitem__(self,index):

        img=self.patchedset[index]
        #label=self.label[index]# if images have different labels
        label=self.label
        
        if self.transforms:
            img=self.transforms(img)
        return (img,label)   
     def patch_creater(self,path,label,patchsize=256):
        self.path=path
        self.label=label
        img = Image.open(self.path)

        patches=np.array([])
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
                    if len(patches.shape) == 1:
                        patches=patchedset
                    else:
                        patches = np.concatenate((patches, patchedset), axis=0)

                    del patchedset

        
        
        
        
class Models(Per_Epoch):
    def __init__(self):
        super().__init__()
        
    def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
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

        # Loop through data loader data batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate  and accumulate loss
            loss = loss_fn(y_pred, y)
            train_loss += loss.item() 

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        # Adjust metrics to get average loss and accuracy per batch 
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        return train_loss, train_acc
    
    def test_step(self,model, 
              testdata,label 
              device)
        model.eval()
        self.y_pred=model(testdata)
        self.y_label=label
        accuracy=self.accuracy_fn()
        cm=self.confusion_matrics()
        f1_score,f2_score=self.f_score()
        balanced_accuracy=self.balanced_accuracy()
        print(f"Accuracy: {accuracy}, Confusion Matrix: {cm}, F1 Score: {f1_score}, F2 Score: {f2_score}, Balanced Accuracy: {balanced_accuracy}")
        clear_output(wait=True)

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
        
        image = np.asarray(img)
        del img
        image_height, image_width, channel_count = image.shape
        patch_height, patch_width, step = patchsize, patchsize, patchsize
        patch_shape = (patch_height, patch_width, channel_count)
        self.patchedset = patchify(image, patch_shape, step=step)
        del image
        gc.collect()
        self.patchedset=self.patchedset.reshape( -1,256, 256, 3)
patched_df=train=pd.read_csv('/kaggle/input/cancerdatasetwithpath/cancerdf.csv')
class trainer(CancerDataSet,Models):
    def __init__(self,patch_size=256):
        self.patch_size=patch_size
        #all hyperparameters
        pass
    def train(self,df):
        for index,row in df.iterrows():
            path=row['Image_path']
            label=row['label']
            dataset=self.patch_creater(path,label,patch_size)
            train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            output=Model()
            
        return 
        #trainmodels 
        
        
        
        

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
            self.train(parameters)
            self.test(parameters)
            
            model.train
        pass
#         return 
        #trainmodels 

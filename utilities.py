import inspect
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn import metrics
from IPython.display import clear_output
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

                                                   
                                                   

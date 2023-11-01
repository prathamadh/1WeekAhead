import inspect
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn import metrics
from IPython.display import clear_output
class HyperParameters:
    def save_hyperparameters(self,ignore=[]):
        '''saves hyperparameters in self.hparams'''
        frame=inspect.currentframe().f_back
        _,_,_,local_vars= inspect.getargvalues(frame)
        self.hparams={k:v for k,v in local_vars.items()
                     if k not in set(ignore+['self']) and not k.startswith('_')}
        
        for k,v in self.hparams.items():
            setattr(self,k,v)
    
class Utilities(HyperParameters):
    def __init__(self):
        pass  
    def _formaty(self,y):
        if len(y.shape)!=1: #one hot encoded 
            y= torch.argmax(y,dim=1)
        return y
    def accuracy_fn(self,y_pred,y_label):
        '''
        format should be for y_label no_of_result*onehot
        and for y_pred it is converted to total_sum*onehot
        '''
       
        y_pred=torch.reshape(y_pred,(-1,y_pred.shape[-1]))#mixes batches 
        y_pred=self._formaty(y_pred)
        y_label=self._formaty(y_label)
        print(y_pred)
        print(y_label)
        correct_pred=torch.sum(y_pred==y_label).item()
        total= y_label.size(0)
        accuracy= correct_pred/total
        return accuracy
    
    def confusion_matrics(self,y_pred,y_label):
        '''plots a confusion matrics and return confusion matrics'''
        y_pred=self._formaty(y_pred)
        y_label=self._formaty(y_label)
        display_labels=None #yet to code
        cm=confusion_matrix(y_label, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()
        return cm
    
    def f_score(self,y_pred,y_label):
        '''returns f1 score and f2 score'''
        y_pred=self._formaty(y_pred)
        y_label=self._formaty(y_label)
        f1_score=metrics.f1_score(y_label,y_pred,average='macro')
        f2_score=metrics.fbeta_score(y_label,y_pred,beta=2,average='macro')
        return f1_score,f2_score
        
    def balanced_accuracy(self,y_pred,y_label,sample_weight=None,adjusted=False):
        y_pred=self._formaty(y_pred)
        y_label=self._formaty(y_label)
        score_result =metrics.balanced_accuracy_score(y_label,y_pred,sample_weight=sample_weight,adjusted=adjusted)
        return score_result
    
    
class Plot_Epoch:
    def __init__(self,checkpoint_path=None):
        self.checkpoint_path=checkpoint_path
        pass
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
    
    def checkpoints(self,model,optimizer,loss,epochs=0):
        if self.checkpoint_path is None:
            raise ValueError("Checkpoint path is not provided")
        
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': resnet_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
        }, checkpoint_path)

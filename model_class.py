import torch

class LogisticRegression:
    def __init__(self):
        self.w=torch.randn(784)*0.01
        self.b=torch.tensor(0.0)
        if torch.cuda.is_available():
            self.w=self.w.to('cuda')
            self.b=self.b.to('cuda')
            
    def sigmoid(self,z):
        return 1/(1+torch.exp(-z))
        
    def BCE_loss(self,X_train,y_train):
        z=(X_train@self.w)+self.b
        y_pred=self.sigmoid(z)
        y_pred=torch.clamp(y_pred,min=1e-7,max=1-1e-7)
        BCEloss=-(y_train*torch.log(y_pred)+(1-y_train)*torch.log(1-(y_pred)))
        loss=BCEloss.mean()
        return loss

    def gradient_descent(self,X_train,y_train,alpha):
        z=(X_train@self.w)+self.b
        y_pred=self.sigmoid(z)
        y_pred=torch.clamp(y_pred,min=1e-7,max=1-1e-7)
        w_grad=(X_train.T)@(y_pred-y_train)
        b_grad=sum(y_pred-y_train)
        self.w-=w_grad*alpha*len(X_train)
        self.b-=b_grad*alpha*len(X_train)
        return self.w,self.b
    
    def predict(self,X_test):
        z=X_test@self.w+self.b
        p=self.sigmoid(z)
        if p>0.5:
            y_pred=1
        else:
            y_pred=0
        return y_pred
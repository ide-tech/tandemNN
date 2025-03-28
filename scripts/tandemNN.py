import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

######################
class MLPRegressor_forward_torch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        #self.fc4 = nn.Linear(10, 10)
        #self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 8)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        #x = torch.tanh(self.fc4(x))
        #x = torch.tanh(self.fc5(x))
        x = self.fc6(x)
        return x
    
    def fit(self, X_df, Y_df):
        X_np = X_df.values
        Y_np = Y_df.values
        X_tensor = torch.FloatTensor(X_np)
        Y_tensor = torch.FloatTensor(Y_np)
        alpha = 0.00001
        criterion = nn.MSELoss()
        losses = []

        optimizer = optim.LBFGS(self.parameters(), max_iter=10000, lr=1)
        def closure():
            optimizer.zero_grad()
            Y_cal_tensor = self(X_tensor)
            loss_Y = criterion(Y_cal_tensor, Y_tensor)

            l2 = torch.tensor(0., requires_grad=True)
            for w in self.parameters():
                l2 = l2 + torch.norm(w)**2

            loss = loss_Y + alpha * l2
            losses.append(loss.item())

            loss.backward()

            return loss
        
        optimizer.step(closure)
        self.losses = losses


    def predict(self, X_df):
        X_np = X_df.values
        X_tensor = torch.FloatTensor(X_np)
        self.eval()
        Y_cal_tensor = self(X_tensor)
        Y_cal_np = Y_cal_tensor.detach().numpy()

        return Y_cal_np
    
######################
class MLPRegressor_inverse_torch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        #self.fc4 = nn.Linear(10, 10)
        #self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 6)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        #x = torch.tanh(self.fc4(x))
        #x = torch.tanh(self.fc5(x))
        x = self.fc6(x)
        return x
    
    def fit(self, X_df, Y_df, model_forward):
        X_np = X_df.values
        Y_np = Y_df.values
        X_tensor = torch.FloatTensor(X_np)
        Y_tensor = torch.FloatTensor(Y_np)
        R_X = 0.1
        R_Y = 1
        alpha = 0.0001
        criterion_X = nn.MSELoss()
        criterion_Y = nn.MSELoss()
        losses = []
        
        optimizer = optim.LBFGS(self.parameters(), max_iter=10000, lr=0.1)
        def closure():
            optimizer.zero_grad()
            X_cal_tensor = self(Y_tensor)
            model_forward.eval()
            Y_cal_tensor = model_forward(X_cal_tensor)
            
            loss_X = criterion_X(X_cal_tensor, X_tensor)
            loss_Y = criterion_Y(Y_cal_tensor, Y_tensor)

            l2 = torch.tensor(0., requires_grad=True)
            for w in self.parameters():
                l2 = l2 + torch.norm(w)**2

            loss = R_X * loss_X + R_Y * loss_Y + alpha * l2
            losses.append(loss.item())

            loss.backward()
            return loss
        
        optimizer.step(closure)
        self.losses = losses
    
    def predict(self, Y_df):
        Y_np = Y_df.values
        Y_tensor = torch.FloatTensor(Y_np)
        self.eval()
        X_cal_tensor = self(Y_tensor)
        X_cal_np = X_cal_tensor.detach().numpy()

        return X_cal_np



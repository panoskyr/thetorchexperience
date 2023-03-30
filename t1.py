import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import numpy as np
import torch 
from torch import nn

from sklearn.model_selection import train_test_split

torch.manual_seed(1955)
n_samples=1000
X,y=make_circles(n_samples, noise=0.03,
                 random_state=1955)


circles=pd.DataFrame(
{
    "X1": [x[0] for x in X ],
    "X2": [x[1] for x in X],
    "label":y
}
)

print(circles.head(5))

print(circles.label.value_counts())
circles.plot(x='X1', y="X2", 
             kind='scatter', 
             c='label' )
#plt.show()


X=torch.from_numpy(X).type(torch.float)
#maybe int?
y=torch.from_numpy(y).type(torch.float)


X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42)
print(len(X_train), len(X_test),len(y_train), len(y_test))



device="cuda" if torch.cuda.is_available() else "cpu"
print(device)


class model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1=nn.Linear(in_features=2,out_features=5)
        self.layer_2=nn.Linear(in_features=5, out_features=1)

    def forward(self,x):
        return self.layer_2(self.layer_1(x))
    
model1_0=model1().to(device)
print(model1_0)

# Make predictions with the model
untrained_preds = model1_0(X_test).to(device)
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{y_test[:10]}")


## this works with raw logits witout passing through 
## the sigmoid.

loss_fn=nn.BCEWithLogitsLoss()
optimizer=torch.optim.SGD(params=model1_0.parameters(), lr=0.1)



def accuracy_fn(y_true,y_pred):
    correct=torch.eq(y_true, y_pred).sum()
    acc=correct/len(y_pred)
    acc=acc*100
    return acc


y_logits=model1_0(X_test).to(device)
y_pred_probs=torch.sigmoid(y_logits)
y_preds=torch.round(y_pred_probs)


y_pred_labels=torch.round(torch.sigmoid(model1_0(X_test).to(device))[:5])
print(y_pred_labels.squeeze())

epochs=50
for epoch in range(epochs):
    ## enable change of gradients
    model1_0.train()

    y_logits=model1_0(X_train).squeeze()
    y_pred=torch.round(torch.sigmoid(y_logits))

    loss=loss_fn(y_logits,y_train)
    acc=accuracy_fn(y_true=y_train, y_pred=y_pred)


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    ## turnoff 
    model1_0.eval()

    ## smae as torch.no_grad()
    with torch.inference_mode():
        test_logits=model1_0(X_test).squeeze()
        test_pred=torch.round(torch.sigmoid(test_logits))

        test_loss=loss_fn(test_logits,y_test)
        test_acc=accuracy_fn(y_true=y_test,
                             y_pred=test_pred)
        
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

import requests 
from pathlib import Path

if Path("helper_functions.py").is_file():
      print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

def plotDecisions():
    plt.figure(figsize=(12,6))
    #nrows #ncols 
    plt.subplot(1,2,1)
    plt.title("Train")
    plot_decision_boundary(model1_0,X_train,y_train)
    plt.subplot(1,2,2)
    plt.title("test")
    plot_decision_boundary(model1_0,X_test,y_test)
    plt.show()
plotDecisions()

## 
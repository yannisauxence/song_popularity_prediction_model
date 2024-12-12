# %% [markdown]
# # Name: YANNIS BOADJI

# %% [markdown]
# #### Project: Song Popularity Prediction

# %%
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("mode.chained_assignment",None)
pd.options.display.max_rows = 5

# %% [markdown]
# ## Data Preparation

# %%
#Loading Dataset

spotify_dataset = pd.read_csv('../data/data.csv')
display(spotify_dataset)

# %%
#Mapping music genres to numbers and replacing them in the dataset

genres = sorted(list(set(spotify_dataset["Genre"])))

character_to_num = { ch:i for i,ch in enumerate(genres) }
print(character_to_num)

spotify_copy = spotify_dataset.copy()
mask = spotify_copy.loc[:,"Genre"]

for i, ch in enumerate(spotify_copy.loc[:,"Genre"]):
    spotify_copy.loc[:,"Genre"][i] = character_to_num[ch]

#Moving column "Genre" to the previous index for easier splitting
col = spotify_copy.pop('Genre')
spotify_copy.insert(16,col.name,col)

display(spotify_copy)

# %%
#Data scaling and Splitting

scaler = StandardScaler()
dataset = spotify_copy.to_numpy()

features = scaler.fit_transform(dataset[-11000:,3:17])
targets = scaler.fit_transform(dataset[-11000:,-1].reshape(11000,1))

train_features = features[-features.shape[0]:-1000]
train_targets = targets[-targets.shape[0]:-1000]
print(train_features.shape,train_targets.shape)

test_features = features[-1000:]
test_targets = targets[-1000:]
print(test_features.shape,test_targets.shape)

# %% [markdown]
# ## Model Definition

# %%
class PopularityMLP(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden1_dim, hidden2_dim, hidden3_dim):
        
        super(PopularityMLP, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden1_dim)
        self.layer2 = torch.nn.Linear(hidden1_dim, hidden2_dim) 
        self.layer3 = torch.nn.Linear(hidden2_dim, hidden3_dim) 
        self.layer4 = torch.nn.Linear(hidden3_dim, output_dim) 
        self.activation = torch.nn.Sigmoid()
    
    def forward(self, x):
        
        out1 = torch.nn.functional.relu(self.layer1(x))
        out2 = torch.nn.functional.relu(self.layer2(out1))
        out3 = torch.nn.functional.relu(self.layer3(out2))
        out4 = self.layer4(out3)
        output = self.activation(out4)
        
        return output

class PopularityRegression(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim): 
        
        super(PopularityRegression, self).__init__() 
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x): 
        
        out = self.linear(x)
        output = self.activation(out)
        
        return output

# %%
torch.manual_seed(55)

model1 = PopularityMLP(input_dim = 14, output_dim = 1, hidden1_dim = 11, hidden2_dim = 7, hidden3_dim=3)
learning_rate1 = 0.001
epochs = 4
batchsize = 100
optimizer1 = torch.optim.Adam(model1.parameters(), lr = learning_rate1)
model1

# %%
model2 = PopularityMLP(input_dim = 14, output_dim = 1, hidden1_dim = 11, hidden2_dim = 7, hidden3_dim=3)
learning_rate2 = 0.001
optimizer2 = torch.optim.Adam(model2.parameters(), lr = learning_rate2)
model2

# %%
model3= PopularityRegression(input_dim = 14, output_dim = 1)
learning_rate3 = 0.01
optimizer3 = torch.optim.Adam(model3.parameters(), lr = learning_rate3)
model3

# %%
#Splitting & Batching

train_features = torch.from_numpy(train_features).float()
train_targets = torch.from_numpy(train_targets).float()
test_features = torch.from_numpy(test_features).float()
test_targets = torch.from_numpy(test_targets).float()

train_batches_features = torch.split(train_features, batchsize)
train_batches_targets = torch.split(train_targets, batchsize)

batch_split_num = len(train_batches_features)

#Training Loss Tracking & Loss Function

train_loss1 = np.zeros((epochs+1,batch_split_num))
train_loss2 = []
train_loss3 = []

loss_func = torch.nn.MSELoss()

# %% [markdown]
# ## Model Training

# %%
import tqdm

#Model 1 Training
print('Model 1  ______________________________________')

for epoch in range(epochs):
    for i in tqdm.trange(batch_split_num):

        optimizer1.zero_grad()

        train_outputs1 = model1(train_batches_features[i])

        loss = loss_func(train_outputs1, train_batches_targets[i])

        train_loss1[epochs][i] = loss.item()

        loss.backward()

        optimizer1.step()
        
    print("Epoch ", epoch,"- Training Loss: ", np.mean(train_loss1[epochs])) #[-batch_split_num:]

#Model 2 Training
print('\nModel 2  ______________________________________')

for epoch in range(int(epochs*10)):
        optimizer2.zero_grad()

        train_outputs2 = model2(train_features)

        loss = loss_func(train_outputs2, train_targets)

        train_loss2.append(loss.item())

        loss.backward()

        optimizer2.step()
        print("Epoch ", epoch,"- Training Loss: ", loss.item())
    
#Model 3 Training
print('\nModel 3  ______________________________________')    

for epoch in range(int(epochs*21)): 

    optimizer3.zero_grad()

    outputs = model3(train_features)

    loss = loss_func(outputs, train_targets) 
    
    train_loss3.append(loss.item()) 
    
    loss.backward() 

    optimizer3.step() 

    print('epoch {}: loss {}'.format(epoch, loss.item()))

# %% [markdown]
# ## Model Evaluation

# %%
# Training Loss Visualization

plt.figure(figsize = (14, 4))

plt.subplot(1, 3, 1)
plt.plot(train_loss1, linewidth = 3, color = 'pink')
plt.ylabel("Model 1 Training Loss")
plt.xlabel("epochs")
sns.despine()

plt.subplot(1, 3, 2)
plt.plot(train_loss2, linewidth = 3, color = 'red')
plt.ylabel("Model 2 Training Loss")
sns.despine()

plt.subplot(1, 3, 3)
plt.plot(train_loss3, linewidth = 3, color = 'blue')
plt.ylabel("Model 3 Training Loss")
sns.despine()

plt.tight_layout()
plt.show()

# %%
#Testing Loss Computation

with torch.no_grad():

        test_outputs1 = model1(test_features)
        test_loss1 = loss_func(test_outputs1, test_targets) 
        print(f'Model 1 Testing MSE Error: '+str(test_loss1.numpy()))

        test_outputs2 = model2(test_features)
        test_loss2 = loss_func(test_outputs2, test_targets) 
        print(f'Model 2 Testing MSE Error: '+str(test_loss2.numpy()))

        test_outputs3 = model3(test_features)
        test_loss3 = loss_func(test_outputs3, test_targets) 
        print(f'Model 3 Testing MSE Error: '+str(test_loss3.numpy()))

# %%
#Scaling Back the data

test1_inv = np.around(scaler.inverse_transform(test_outputs1))
test2_inv = np.around(scaler.inverse_transform(test_outputs2))
test3_inv = np.around(scaler.inverse_transform(test_outputs3))
test_inv = np.around(scaler.inverse_transform(test_targets))

# %%
#Testing Accuracy

#Correct Predictions

n1 = 0
n2 = 0
n3 = 0

for i in range(test_inv.shape[0]):
    if test_inv[i] == test1_inv[i]:
        n1 += 1
    elif test_inv[i] == test2_inv[i]:  
        n2 += 1
    elif test_inv[i] == test3_inv[i]:
        n3 += 1

#Visualization
plt.figure(figsize = (12, 4))

plt.subplot(1,3,1)
plt.plot(test_inv, label='Ground Truth')
plt.plot(test1_inv, label='Model 1 Predictions')
plt.xlabel('Model 1 Correct Predicitions : '+str(n1)+'/1000')
plt.legend()
sns.despine()

plt.subplot(1,3,2)
plt.plot(test_inv, label='Ground Truth')
plt.plot(test2_inv, label='Model 2 Predictions')
plt.xlabel('Model 2 Correct Predictions : '+str(n2)+'/1000')
plt.legend()
sns.despine()

plt.subplot(1,3,3)
plt.plot(test_inv, label='Ground Truth')
plt.plot(test3_inv, label='Model 3 Predictions')
plt.xlabel('Model 3 Correct Predictions : '+str(n3)+'/1000')
plt.legend()
sns.despine()

plt.tight_layout()
plt.show()  

# %%
#Model Saving

torch.save({
            'model1_state_dict': model1.state_dict(),
            'model2_state_dict': model2.state_dict(),
            'model3_state_dict': model3.state_dict(),
            'optimizer1_state_dict': optimizer1.state_dict(),
            'optimizer2_state_dict': optimizer2.state_dict(),
            'optimizer3_state_dict': optimizer3.state_dict(),
            },'../checkpoint/models.tar' )

testing = np.save('../demo/spotify_testing_data',test_features)

# %%
print(test_features[0])



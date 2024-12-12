import torch


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
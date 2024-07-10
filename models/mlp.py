import torch
import torch.nn as nn
import torch.optim as optim

class MLPModel(nn.Module):
    def __init__(self, config):
        super(MLPModel, self).__init__()
        layers = []
        input_dim = config["input_dim"]
        for hidden_dim in config["hidden_layers"]:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, config["output_dim"]))
        self.network = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        return self.network(x)

    def predict(self, state):
        state = torch.FloatTensor(state).view(-1, self.network[0].in_features).to(self.device)
        return self.forward(state)

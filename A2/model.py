import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, state_size: int = 4, action_size: int = 1):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.net = nn.Sequential(
            nn.Linear(self.state_size, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),      
            nn.Linear(32, self.action_size),
        )
        # TODO YOUR CODE HERE FOR INITIALIZING THE MODEL
        # Guidelines for network size: start with 2 hidden layers and maximum 32 neurons per layer
        # feel free to explore different sizes

    def forward(self, x):
        # TODO YOUR CODE HERE FOR THE FORWARD PASS
        return self.net(x).view(-1, self.action_size)
    
    def select_action(self, state):
        self.eval()
        x = self(state)
        self.train()
        return x.max(1)[1].view(1, 1).to(torch.long)

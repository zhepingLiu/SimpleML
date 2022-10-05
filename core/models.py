import torch

class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, bias=True):
        super(SimpleNN, self).__init__()
        
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=bias),
            # torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size, bias=bias)
        )
    
    def forward(self, input):
        return self.seq(input)
    
class MyNN1(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, bias=True):
        super(MyNN1, self).__init__()
        
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size*2, bias=bias),
            # TODO: don't know why BN layer work poorly during validation
            # torch.nn.BatchNorm1d(hidden_size*2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size*2, hidden_size, bias=bias),
            # torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size, bias=bias)
        )
    
    def forward(self, input):
        return self.seq(input)
    
class MyNN2(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, bias=True):
        super(MyNN2, self).__init__()
        
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=bias),
            # torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=bias),
            # torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size, bias=bias)
        )
    
    def forward(self, input):
        return self.seq(input)
  
class MyCNN1(torch.nn.Module):
    def __init__(self, input_channel, output_size, hidden_size=128):
        super(MyCNN1, self).__init__()
        
        self.seq = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=input_channel, out_channels=5, kernel_size=3, stride=1, padding=1, 
                            dilation=1, padding_mode='zeros', bias=True),
            # torch.nn.Conv1d(in_channels=input_channel, out_channels=5, kernel_size=4, stride=1, padding=3, 
            #                 dilation=2, padding_mode='zeros', bias=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=None, padding=0, dilation=1),
            torch.nn.ReLU()
        )
        
        self.seq2= torch.nn.Sequential(
            torch.nn.Linear(1, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, input):
        input = input.reshape(-1, 1, 12)
        res = self.seq(input)
        res = res.reshape(-1, 1)
        return self.seq2(res)
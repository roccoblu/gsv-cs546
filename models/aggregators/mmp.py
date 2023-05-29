import torch
import torch.nn.functional as F
import torch.nn as nn

# class MeanMaxPool(nn.Module):
#     """
#     Implementation of MeanMax Pooling as described in the given equations.
#     """
#     def __init__(self, in_channels, out_channels):
#         super(MeanMaxPool, self).__init__()
#         self.linearTransform = nn.Linear(2*in_channels, out_channels)
#         self.activation = nn.Tanh()

    # def forward(self, hidden_states):
    #   mag = torch.norm(hidden_states, dim = 2)
    #   max_idx = torch.argmax(mag, dim = 1)
    #   max_T = torch.Tensor()
    #   max_T = max_T.to('cuda')

    #   for i,idx in enumerate(max_idx):
    #     h = hidden_states[i,idx,:]
    #     max_T = torch.cat((max_T,h.view(-1, len(h))))

    #   sum_T = torch.squeeze(torch.sum(hidden_states, dim = 1))
    #   C_mmt = torch.cat((sum_T, max_T),dim=1)

    #   pooled_output = self.activation(self.linearTransform(C_mmt))
      
    #   # Normalize
    #   C = F.normalize(pooled_output, p=2, dim=1)

    #   return C
import torch.nn as nn
import torch.nn.functional as F

class MeanMaxPool(nn.Module):
    """
    Implementation of MeanMax Pooling as described in the given equations.
    """
    
    def __init__(self, in_channels, out_channels, height, width):
        super(MeanMaxPool, self).__init__()
        self.linearTransform = nn.Linear(2*in_channels*height*width, out_channels)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # Flatten the spatial dimensions
        batch_size, channels, height, width = hidden_states.size()
        hidden_states = hidden_states.view(batch_size, channels, -1)

        # Calculate magnitude of each feature vector
        mag = torch.norm(hidden_states, dim=1, keepdim=True)

        # Calculate max token
        _, max_indices = torch.max(mag, dim=2, keepdim=True)
        max_T = torch.gather(hidden_states, 2, max_indices)

        # Calculate sum of tokens
        sum_T = torch.sum(hidden_states, dim=2, keepdim=True)

        # Concatenate sum and max tokens
        C_mmt = torch.cat((sum_T, max_T), dim=1)

        # Apply linear transformation and tanh activation
        pooled_output = self.activation(self.linearTransform(C_mmt.view(batch_size, -1)))

        # Normalize
        C = F.normalize(pooled_output, p=2, dim=1)

        return C


if __name__ == '__main__':
    hidden_states = torch.randn(4, 2048, 10, 10)  # Batch Size x Maximum Sequence Length x Hidden Size
    m = MeanMaxPool(2048, 512, 10, 10)  # Input Channels = Hidden Size, Output Channels = 512, height = 10, width = 10
    r = m(hidden_states)
    print(r.shape)  # Should print torch.Size([4, 512])

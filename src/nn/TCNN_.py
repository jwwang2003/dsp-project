import torch
import torch.nn as nn

# Our Temporal Neural Network Implementation

class ResidualConvBlock(nn.Module):
  def __init__(self, input_channels, output_channels, kernel_size, dropout):
    super(ResidualConvBlock, self).__init__()
    self.conv = nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)
      
    # Optional: Adjust for residual connection if input and output channels are different
    if input_channels != output_channels:
      self.shortcut = nn.Conv1d(input_channels, output_channels, kernel_size=1)  # Match the dimensions
    else:
      self.shortcut = nn.Identity()  # No adjustment needed

  def forward(self, x):
      residual = self.shortcut(x)  # Adjust the residual connection if needed
      out = self.conv(x)
      out = self.relu(out)
      out = self.dropout(out)
      out += residual  # Add the adjusted residual
      return out

class BiTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, seq_len, dropout):
      super(BiTCN, self).__init__()
      self.tcn_layers = nn.ModuleList()
      
      # Create multiple residual blocks
      for i in range(len(num_channels)):
        input_channels = input_size if i == 0 else num_channels[i-1]
        self.tcn_layers.append(ResidualConvBlock(input_channels, num_channels[i], kernel_size, dropout))
      
      # Final linear layer to output size
      self.linear = nn.Linear(num_channels[-1] * seq_len, output_size)

    def forward(self, inputs):
      """Inputs have to have dimension (N, C_in, L_in)"""
      # Ensure inputs remain in the shape (N, C_in, L_in)
      for layer in self.tcn_layers:
        inputs = layer(inputs)  # Pass through each residual block
      
      # Flatten the output before the linear layer
      y_flat = inputs.reshape(inputs.shape[0], -1)
      o = self.linear(y_flat)
      return o


# Example usage
if __name__ == "__main__":
    input_size = 1  # Number of input channels
    output_size = 1  # Number of output channels (for regression)
    seq_length = 45  # Length of the input sequence
    batch_size = 32  # Example batch size
    # num_channels = [20, 40, 60, 80, 100]  # Number of output channels for each layer
    num_channels = [20] * 5

    # Create a random input tensor (batch_size, input_channels, seq_length)
    input_tensor = torch.rand(batch_size, input_size, seq_length)
    
    # Create BiTCN model
    model = BiTCN(input_size, output_size, num_channels, kernel_size=3, seq_len=seq_length, dropout=0.2)
    
    print(model)
    # Forward pass
    output = model(input_tensor)
    
    print("Output shape:", output.shape)  # Should be (batch_size, output_size)
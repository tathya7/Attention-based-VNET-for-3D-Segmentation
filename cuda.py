import torch
import torch.nn as nn

# Assuming unlab_ema_out is a list of tensors
unlab_ema_out = [
    torch.randn(2, 3, 64, 64, 64),
    torch.randn(2, 3, 64, 64, 64),
    torch.randn(2, 3, 64, 64, 64),
    torch.randn(2, 128, 64, 64, 64)
]

# Check the shape of the tensors
print(f"Shape of unlab_ema_out[3] before: {unlab_ema_out[3].shape}")

# Option 1: Select the first 3 channels (if appropriate)
unlab_ema_out[3] = unlab_ema_out[3][:, :3, :, :, :]

# Option 2: Average the channels to get the first 3 channels
# unlab_ema_out[3] = torch.mean(unlab_ema_out[3], dim=1, keepdim=True).repeat(1, 3, 1, 1, 1)

# Option 3: Using a convolutional layer to reduce channels
# conv = nn.Conv3d(128, 3, kernel_size=1)
# unlab_ema_out[3] = conv(unlab_ema_out[3])

# Check the shape of the tensors again
print(f"Shape of unlab_ema_out[3] after: {unlab_ema_out[3].shape}")

# Now you can safely sum the tensors
unlab_ema_out_pred = (unlab_ema_out[0] + unlab_ema_out[1] + unlab_ema_out[2] + unlab_ema_out[3]) / 4

import torch
import numpy as np
import h5py
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from networks.vnet_AMC import VNet_AMC

def load_h5_data(h5_path):
    with h5py.File(h5_path, 'r') as h5_file:
        image_data = h5_file['image'][()]
    return image_data

def save_predictions(model, image_data, save_path):
    model.eval()
    with torch.no_grad():
        image_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        predictions = model(image_tensor)
        
    
        if isinstance(predictions, list):
            print(f"Model returned {len(predictions)} outputs.")
            for i, pred in enumerate(predictions):
                print(f"Output {i}: shape {pred.shape}")
            
            predictions = predictions[0]
        
        predictions = predictions.argmax(dim=1).squeeze().cpu().numpy()
    np.save(save_path, predictions)
    print(f"Predictions saved to {save_path}")

model_path = '/home/amishr17/aryan/new_attempt/results/TBAD/checkpoints/best.pth'
checkpoint = torch.load(model_path)
model_state_dict = checkpoint['net']


new_state_dict = {}
for k, v in model_state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v


model = VNet_AMC(n_channels=1, n_classes=3, n_branches=4)
model.load_state_dict(new_state_dict)
model.cuda()


image_path = '/home/amishr17/aryan/new_attempt/preprocess/TBAD/ImageTBAD/TBAD-40.h5'
image_data = load_h5_data(image_path)


save_dir = 'results/vis/TBAD_45/'
os.makedirs(save_dir, exist_ok=True)

# Generate and save predictions
save_path = os.path.join(save_dir, 'UPCoL.npy')
save_predictions(model, image_data, save_path)

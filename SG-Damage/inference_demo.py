import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.lstm import LSTM200
from model.cgan import Generator
from model.fpn import FPN

# ======= Device =======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======= Paths =======
csv_path = 'data/demo.csv'
pt_path = 'data/demo.pt'
weight_paths = {
    'LSTM200': 'weight/batch_663000.pth',
    'Generator': 'weight/395500_generator.pth',
    'FPN': 'weight/755200.pth'
}

# ======= Utility: Safe Load =======
def safe_load_model(model, weight_path, model_name):
    if model_name == "Generator":
        state_dict = torch.load(weight_path, map_location=device)["state_dict"]
    else:
        state_dict = torch.load(weight_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
        print(f"[{model_name}] Loaded successfully.")
    except RuntimeError as e:
        print(f"[{model_name}] Load failed!")
        print(e)
        exit(1)
    return model

# ======= 1. Load Sensor Data =======
sensor_df = pd.read_csv(csv_path, header=None)
sensor_df = sensor_df.apply(pd.to_numeric, errors='coerce')
if sensor_df.isnull().any().any():
    print("Warning: Non-numeric values detected in CSV. Filling with 0.")
    sensor_df = sensor_df.fillna(0)

sensor = sensor_df.values.astype(np.float32).squeeze()
sensor = torch.tensor(sensor, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 200)

# ======= 2. Load Ground Truth =======
strain_gt = torch.load(pt_path).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 151, 151)

# ======= 3. LSTM Interpolation =======
lstm = LSTM200().to(device)
lstm = safe_load_model(lstm, weight_paths['LSTM200'], 'LSTM200')
lstm.eval()
with torch.no_grad():
    interp_seq = lstm(sensor)  # (1, 200)

# ======= 4. Generator: Strain Reconstruction =======
generator = Generator().to(device)
generator = safe_load_model(generator, weight_paths['Generator'], 'Generator')
generator.eval()
with torch.no_grad():
    gen_strain = generator(interp_seq)  # (1, 1, 151, 151)

# ======= 5. Resize to 600x600 =======
def resize_to_600(x):
    return F.interpolate(x, size=(600, 600), mode='bilinear', align_corners=False)

gen_strain_600 = resize_to_600(gen_strain)

# ======= 6. FPN Coordinate Recognition =======
fpn = FPN().to(device)
fpn = safe_load_model(fpn, weight_paths['FPN'], 'FPN')
fpn.eval()
with torch.no_grad():
    pred_coords = fpn(gen_strain_600)  # (1, 2, 600, 600)
    pred_coords_sigmoid = torch.sigmoid(pred_coords)
    heatmap = pred_coords_sigmoid.sum(dim=1, keepdim=True)  # shape: (1, 1, 600, 600)

# ======= 7. Visualization =======
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].plot(sensor.cpu().numpy().squeeze())
axes[0].set_title("Input Sensor (200)")
axes[1].imshow(gen_strain[0, 0].cpu(), cmap='hsv')
axes[1].set_title("Generated Strain (151x151)")
axes[2].imshow(gen_strain_600[0, 0].cpu(), cmap='hsv')
axes[2].set_title("Upscaled Strain (600x600)")
axes[3].imshow(heatmap[0, 0].cpu(), cmap='hot')
axes[3].set_title("Summed Coord Heatmap (2 channels)")
plt.tight_layout()
plt.savefig("output_inference.png", dpi=300)
plt.show()
print("Inference complete. Output saved to output_inference.png")

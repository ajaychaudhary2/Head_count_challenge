import torch
import yaml
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load config
with open(r'G:\head_count\head_count challenge\Congfig\config (1).yaml') as f:
    config = yaml.safe_load(f)

# Load model
model = torch.load(r'G:\head_count\head_count challenge\Model\model_final.pth')
model.eval()

# Image path for testing
image_path = r'G:\head_count\head_count challenge\Data\sample.jpg'
image = Image.open(image_path).convert('RGB')

# Preprocess (depends on your config)
transform = transforms.Compose([
    transforms.Resize((config['input_size'], config['input_size'])),
    transforms.ToTensor()
])

input_tensor = transform(image).unsqueeze(0)  # add batch dim

# Predict
with torch.no_grad():
    output = model(input_tensor)
    predicted_count = output.item()

# Show image + prediction
plt.imshow(image)
plt.title(f'Predicted Head Count: {predicted_count:.2f}')
plt.axis('off')
plt.show()

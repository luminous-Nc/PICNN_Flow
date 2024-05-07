import torch
from torchvision import transforms
from PIL import Image
from model.new_model import PICNN
import time

# 记录推断开始时间
start_time = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PICNN().to(device)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

input_image = Image.open('images/input.png')

preprocess = transforms.Compose([
    transforms.ToTensor()
])
input_tensor = preprocess(input_image).to(device)

if input_tensor.shape[0] == 4:
    input_tensor = input_tensor[1:]

input_tensor = input_tensor.unsqueeze(0)

with torch.no_grad():
    output_a,output_b = model(input_tensor)


output_a_image = transforms.ToPILImage()(output_a[0].cpu())
output_b_image = transforms.ToPILImage()(output_b[0].cpu())


output_a_image.save('images/output_x.png')
output_b_image.save('images/output_y.png')




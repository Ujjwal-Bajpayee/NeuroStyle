import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import copy
from io import BytesIO

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
imsize = 512 if torch.cuda.is_available() else 256
loader = transforms.Compose([transforms.Resize((imsize, imsize)), transforms.ToTensor()])

def image_loader(image):
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Loss modules
class ContentLoss(nn.Module):
    def __init__(self, target): super().__init__(); self.target = target.detach()
    def forward(self, x): self.loss = nn.functional.mse_loss(x, self.target); return x

def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature): super().__init__(); self.target = gram_matrix(target_feature).detach()
    def forward(self, x): G = gram_matrix(x); self.loss = nn.functional.mse_loss(G, self.target); return x

class Normalization(nn.Module):
    def __init__(self, mean, std): super().__init__(); self.mean = mean.view(-1, 1, 1); self.std = std.view(-1, 1, 1)
    def forward(self, img): return (img - self.mean) / self.std

# Build model
def get_style_model_and_losses(cnn, norm_mean, norm_std, style_img, content_img,
                                content_layers=['conv_4'],
                                style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(norm_mean, norm_std).to(device)
    content_losses, style_losses = [], []
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d): i += 1; name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU): name = f'relu_{i}'; layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d): name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d): name = f'bn_{i}'
        else: continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_" + name, content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module("style_loss_" + name, style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss): break
    return model[:(i+1)], style_losses, content_losses

# Style Transfer
def run_style_transfer(cnn, norm_mean, norm_std, content_img, style_img, input_img,
                       num_steps=300, style_weight=1000000, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, norm_mean, norm_std, style_img, content_img)
    optimizer = torch.optim.LBFGS([input_img.requires_grad_()])
    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum([sl.loss for sl in style_losses])
            content_score = sum([cl.loss for cl in content_losses])
            loss = style_weight * style_score + content_weight * content_score
            loss.backward(); run[0] += 1
            return loss
        optimizer.step(closure)
    input_img.data.clamp_(0, 1)
    return input_img

# Load model
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Streamlit UI
st.set_page_config(layout="centered")
st.title("🎨 Neural Style Transfer")

use_sample = st.toggle("🖼 Use Sample Images")

if use_sample:
    content_image = Image.open("image.jpg").convert("RGB")
    style_image = Image.open("demo.jpg").convert("RGB")
else:
    content_file = st.file_uploader("Upload a content image", type=["jpg", "jpeg", "png"])
    style_file = st.file_uploader("Upload a style image", type=["jpg", "jpeg", "png"])
    if content_file and style_file:
        content_image = Image.open(content_file).convert("RGB")
        style_image = Image.open(style_file).convert("RGB")
    else:
        st.stop()

col1, col2 = st.columns(2)
col1.image(content_image, caption="🖼️ Content Image", use_container_width=True)
col2.image(style_image, caption="🎨 Style Image", use_container_width=True)

# Simpler UI slider from 1 to 100, but scaled internally
style_slider = st.slider("🎛️ Style Strength (1–100)", 1, 100, 60)
style_weight = style_slider * 100000  # Converts 1–100 to 100K–10M
st.write("This value controls how much of the style is applied to the content image.")
st.write("Higher values will apply more style, while lower values will keep more of the content.")

if st.button("✨ Stylize"):
    content_tensor = image_loader(content_image)
    style_tensor = image_loader(style_image)
    input_tensor = content_tensor.clone()

    with st.spinner("Stylizing... please wait..."):
        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    content_tensor, style_tensor, input_tensor,
                                    style_weight=style_weight, content_weight=1)

    output_image = transforms.ToPILImage()(output.squeeze().cpu())
    st.image(output_image, caption="🖌️ Stylized Output", use_container_width=True)

    buf = BytesIO()
    output_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button("📥 Download Stylized Image", byte_im, file_name="stylized_output.png", mime="image/png")
    st.success("Done!")



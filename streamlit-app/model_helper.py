import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os

trained_model = None

class_names = [
    'Front Breakage',
    'Front Crushed',
    'Front Normal',
    'Rear Breakage',
    'Rear Crushed',
    'Rear Normal'
]


class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.model = models.resnet50(weights='DEFAULT')

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last block
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def load_model():
    global trained_model

    if trained_model is None:
        # Get absolute path of current file
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Build full path to model file
        model_path = os.path.join(base_dir, "model", "saved_model.pth")

        trained_model = CarClassifierResNet()
        trained_model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        trained_model.eval()

    return trained_model


def predict(image_path):
    model = load_model()

    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)

    return class_names[predicted_class.item()]
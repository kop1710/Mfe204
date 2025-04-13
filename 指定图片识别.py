import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet18
# Define the corrected StudentCNN class
class StudentCNN(nn.Module):
    def __init__(self):
        super(StudentCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        feat = self.features(x)
        feat = self.avgpool(feat)
        out = feat.view(feat.size(0), -1)
        out = self.classifier(out)
        return out

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 2. 加载保存的模型
teacher = resnet18(pretrained=False)  # Use ResNet18, but for CIFAR-10 we need to adjust it
teacher.fc = nn.Linear(512, 10)  # CIFAR-10 -> 10 classes for different objects
teacher.to(device)
teacher.load_state_dict(torch.load("/tmp/teacher_kd.pth"))
teacher.eval()  # Set the model to evaluation mode

# 3. Initialize student model
student = StudentCNN().to(device)  # Use the same CNN model as before for simplicity
student.load_state_dict(torch.load(r"C:\Users\Administrator\Desktop\model\best_student.pth"))
student.eval()  # Set the model to evaluation mode

# 4. Preprocessing for the custom image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(32),  # Resize images to 32x32 to match ResNet18 input size
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 normalization
    ])

    # Open the image file
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img

# 5. Make a prediction on a custom image
def predict_custom_image(image_path):
    # Preprocess the custom image
    img = preprocess_image(image_path).to(device)

    # Make predictions with both teacher and student models
    with torch.no_grad():
        teacher_out = teacher(img)
        student_out = student(img)

        teacher_pred = teacher_out.argmax(dim=1).item()
        student_pred = student_out.argmax(dim=1).item()

    # Display the image and predictions
    img_display = Image.open(image_path)
    img_display = img_display.resize((128, 128))  # Resize for better display

    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Display the image and predicted labels
    plt.imshow(img_display)
    plt.title(f"Teacher Prediction: {classes[teacher_pred]}\nStudent Prediction: {classes[student_pred]}")
    plt.axis("off")
    plt.show()

    print(f"Teacher Prediction: {classes[teacher_pred]}")
    print(f"Student Prediction: {classes[student_pred]}")

# 6. Test the function with your custom image
custom_image_path = r"C:\Users\Administrator\Desktop\testd3.jpg"  # Use raw string or double backslashes
predict_custom_image(custom_image_path)

import torch
import torch.nn as nn  # Ensure nn is imported
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import random
import numpy as np
from torchvision.models import resnet18

# Define the StudentCNN class if it is not defined elsewhere in the code
class StudentCNN(nn.Module):
    def __init__(self):
        super(StudentCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        feat = self.features(x)
        out = feat.view(feat.size(0), -1)
        out = self.classifier(out)
        return out

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载保存的模型
teacher = resnet18(pretrained=False)  # Ensure resnet18 is imported
teacher.fc = nn.Linear(512, 10)  # CIFAR-10 -> 10分类
teacher.to(device)
teacher.load_state_dict(torch.load("/tmp/teacher_kd.pth"))
teacher.eval()  # 设置为评估模式

# 3. Initialize student model
student = StudentCNN().to(device)  # Ensure StudentCNN class is defined above
student.load_state_dict(torch.load("/tmp/student_kd.pth"))
student.eval()  # 设置为评估模式

# 4. 加载测试数据集
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

testset = datasets.CIFAR10(root='/tmp/data', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

# 5. Define the class names for CIFAR-10
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 6. 随机图像预测可视化
def visualize_predictions(t_model, s_model, loader, num_images=5):
    t_model.eval()
    s_model.eval()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    idxs = random.sample(range(len(images)), num_images)
    plt.figure(figsize=(15,3))

    for i, idx in enumerate(idxs):
        img = images[idx].cpu().clone()  # Move the image to CPU
        label = labels[idx].item()

        with torch.no_grad():
            img = img.to(device)
            t_out = t_model(img.unsqueeze(0))
            s_out = s_model(img.unsqueeze(0))
            t_pred = t_out.argmax(dim=1).item()
            s_pred = s_out.argmax(dim=1).item()

        img = img.permute(1,2,0)
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        
        # Move the image tensor to CPU and convert to numpy array
        img = img.cpu().numpy() * std + mean
        img = np.clip(img, 0, 1)

        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.title(f"True:{classes[label]}\nTeacher:{classes[t_pred]}\nStudent:{classes[s_pred]}")
        plt.axis("off")
    plt.show()

# 7. 可视化预测
print("\nVisualizing random predictions...")
visualize_predictions(teacher, student, testloader, num_images=5)

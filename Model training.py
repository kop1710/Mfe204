import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


batch_size = 128
epochs_teacher = 20   # 增加教师训练轮数
epochs_student = 20   # 增加学生训练轮数
alpha = 0.7           # 调整软标签权重
T = 2.0               # 降低初始温度
T_min = 1.0           # 最低温度


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# 数据加载
trainset = torchvision.datasets.CIFAR10(
    root='/tmp/data', train=True, download=True, transform=train_transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
)

testset = torchvision.datasets.CIFAR10(
    root='/tmp/data', train=False, download=True, transform=test_transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
)


class EnhancedStudentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


teacher = resnet18(pretrained=False)  
teacher.fc = nn.Linear(512, 10)
teacher.to(device)

optimizer_teacher = optim.Adam(teacher.parameters(), lr=0.001)
scheduler_teacher = optim.lr_scheduler.StepLR(optimizer_teacher, step_size=10, gamma=0.1)


def train_teacher_model(model, loader, epochs):
    model.train()
    for epoch in range(epochs):
        correct, total = 0, 0
        running_loss = 0.0
        
        for inputs, labels in tqdm(loader, desc=f"Teacher Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer_teacher.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer_teacher.step()
            
            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        
        scheduler_teacher.step()
        epoch_acc = 100.0 * correct / total
        print(f"[Teacher] Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(loader):.4f} | Acc: {epoch_acc:.2f}%")
    return model


def train_student_kd(teacher_model, student_model, train_loader, epochs):
    optimizer = optim.Adam(student_model.parameters(), lr=0.0005) 
    best_acc = 0.0
    
    for epoch in range(epochs):
        student_model.train()
        current_T = max(T_min, T - (T - T_min) * (epoch / epochs))  
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Student Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
     
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
            student_logits = student_model(inputs)
            
         
            loss_ce = F.cross_entropy(student_logits, labels)
            loss_kd = F.kl_div(
                F.log_softmax(student_logits/current_T, dim=1),
                F.softmax(teacher_logits/current_T, dim=1),
                reduction='batchmean'
            ) * (current_T ** 2)
            loss = (1 - alpha) * loss_ce + alpha * loss_kd
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = student_logits.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        
        epoch_acc = 100.0 * correct / total
        print(f"[Student] Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Acc: {epoch_acc:.2f}% | T={current_T:.2f}")
        
       
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(student_model.state_dict(), "best_student.pth")
    
    return student_model


if __name__ == "__main__":
   
    print("==== Training Teacher Model ====")
    teacher = train_teacher_model(teacher, trainloader, epochs_teacher)
    teacher.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for inputs, labels in testloader:
            outputs = teacher(inputs.to(device))
            _, preds = outputs.max(1)
            correct += preds.cpu().eq(labels).sum().item()
            total += labels.size(0)
        print(f"[Teacher] Test Accuracy: {100.0*correct/total:.2f}%")


    print("\n==== Training Student Model ====")
    student = EnhancedStudentCNN().to(device)
    student = train_student_kd(teacher, student, trainloader, epochs_student)
    
   
    student.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for inputs, labels in testloader:
            outputs = student(inputs.to(device))
            _, preds = outputs.max(1)
            correct += preds.cpu().eq(labels).sum().item()
            total += labels.size(0)
        print(f"[Student] Final Test Accuracy: {100.0*correct/total:.2f}%")

  
    def get_model_size(model):
        torch.save(model.state_dict(), "temp.pth")
        size = os.path.getsize("temp.pth")/1024**2
        os.remove("temp.pth")
        return size
    
    print(f"\nTeacher Size: {get_model_size(teacher):.2f} MB")
    print(f"Student Size: {get_model_size(student):.2f} MB")




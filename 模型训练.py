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
from tqdm import tqdm  # 导入tqdm


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
epochs_teacher = 20   
epochs_student = 20   
alpha = 0.7           
T = 2.0              
T_min = 1.0           



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

classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {device}")


print(f"Training dataset size: {len(trainloader.dataset)}")
print(f"Test dataset size: {len(testloader.dataset)}")


teacher = resnet18(pretrained=True)
teacher.fc = nn.Linear(512, 10)  # CIFAR-10 -> 10分类
teacher.to(device)

optimizer_teacher = optim.Adam(teacher.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  



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


student = StudentCNN().to(device)
optimizer_student = optim.Adam(student.parameters(), lr=0.001)



def train_teacher_model(model, loader, epochs):
    acc_list = []
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        running_loss = 0.0
        
        print(f"Starting Epoch {epoch+1}/{epochs}")

        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_teacher.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_teacher.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(loader)} | Loss={loss.item():.4f}")

        acc = 100.0 * correct / total
        acc_list.append(acc)
        print(f"[Teacher] Epoch {epoch+1}/{epochs} | Loss={running_loss/len(loader):.3f} | TrainAcc={acc:.2f}%")
    
    return acc_list



def distillation_loss(student_logits, teacher_logits, T):
    """
    KL散度蒸馏损失
    """
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)



def train_student_kd(teacher_model, student_model, loader, epochs, T):
    teacher_model.eval()
    train_acc_list = []

    for epoch in range(epochs):
        student_model.train()
        correct, total = 0, 0
        running_loss = 0.0

  
        for i, (inputs, labels) in enumerate(tqdm(loader, desc=f"Student Epoch {epoch+1}/{epochs}", ncols=100)):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer_student.zero_grad()

           
            student_logits = student_model(inputs)

         
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)

           
            loss_ce = criterion(student_logits, labels)
      
            loss_kd = distillation_loss(student_logits, teacher_logits, T)

            loss = alpha * loss_kd + (1 - alpha) * loss_ce

            loss.backward()
            optimizer_student.step()

            running_loss += loss.item()  
            _, preds = student_logits.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        
            if i % 10 == 0:
                print(f"[Student-KD] Batch {i}/{len(loader)} | Loss={loss.item():.4f}")  

      
        T = max(T_min, T - (T - T_min) * 0.1)  

        acc = 100.0 * correct / total
        train_acc_list.append(acc)  
        print(f"[Student-KD] Epoch {epoch+1}/{epochs} | Loss={running_loss/len(loader):.3f} | TrainAcc={acc:.2f}%")

    return train_acc_list

def evaluate(model, loader):
    model.eval()  # Set the model to evaluation mode
    correct, total = 0, 0
    running_loss = 0.0

    with torch.no_grad():  # Disable gradient tracking
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / total
    print(f"Test Loss={running_loss/len(loader):.3f} | Test Accuracy={acc:.2f}%")
    return acc


print("==== Training Teacher Model ====")
teacher_train_acc = train_teacher_model(teacher, trainloader, epochs_teacher)
teacher_test_acc = evaluate(teacher, testloader)
print(f"Teacher Test Accuracy: {teacher_test_acc:.2f}%")

print("\n==== Training Student Model with Logits Distillation ====")
student_train_acc = train_student_kd(teacher, student, trainloader, epochs_student, T)

student_test_acc = evaluate(student, testloader)  
print(f"\n[Student] Test Accuracy: {student_test_acc:.2f}%")



def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

teacher_params = count_params(teacher)
student_params = count_params(student)

print(f"\nTeacher Params: {teacher_params}")
print(f"Student Params: {student_params}")

teacher_save_path = "/tmp/teacher_kd.pth"
student_save_path = "/tmp/student_kd.pth"  

torch.save(teacher.state_dict(), teacher_save_path)
torch.save(student.state_dict(), student_save_path)

teacher_size = os.path.getsize(teacher_save_path) / 1024**2
student_size = os.path.getsize(student_save_path) / 1024**2
print(f"Teacher Model File Size: {teacher_size:.2f} MB")
print(f"Student Model File Size: {student_size:.2f} MB")



epochs_range_teacher = range(1, epochs_teacher+1)  
epochs_range_student = range(1, epochs_student+1)

plt.figure(figsize=(8, 4))
plt.plot(epochs_range_teacher, teacher_train_acc, label='Teacher Train Acc')
plt.plot(epochs_range_student, student_train_acc, label='Student Train Acc')
plt.title("Training Accuracy (Teacher vs Student-KD)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()




import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='elastic-training-executor/', 
        train=True, download=True, transform=transform)
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

BATCH_SIZE = 60

transform = transforms.Compose([


    transforms.Resize((256,
                        256)),
    transforms.ToTensor(),
])

train_path = 'archive(1)/train'
test_path = 'archive(1)/test'

train_set = ImageFolder(root=train_path,transform=transform)
test_set = ImageFolder(root=test_path,transform=transform)


train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)
test_test_loader = DataLoader(test_set,batch_size=1,shuffle=True)
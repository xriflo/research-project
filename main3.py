#from folder import ImageFolder as standarad
#from dataloader import DataLoader as stdloader
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#from activityfolder import ActivityImageFolder

#print("mama")
'''
std = datasets.ImageFolder("../raw_train_resized")
std_loader = torch.utils.data.DataLoader(
    std, batch_size=1, shuffle=True, num_workers=1)


for batch_idx, (inputs, targets) in enumerate(std_loader):
	print (inputs, targets)

#act = ActivityImageFolder(root="../../activity/raw_activity_small")
'''

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.32950327, 0.32950327, 0.32950327), (0.2481389, 0.2481389, 0.2481389)),
])


std = datasets.ImageFolder("../raw_train_resized", transform_train)
std_loader = torch.utils.data.DataLoader(std, batch_size=1, shuffle=True, num_workers=1)

for x in enumerate(std_loader):
	pass


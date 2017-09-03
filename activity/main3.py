#from folder import ImageFolder as standarad
#from dataloader import DataLoader as stdloader
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from activityfolder import ActivityImageFolder
from activitydataloader import ActivityDataLoader


#from activityfolder import ActivityImageFolder

#print("mama")

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.32950327, 0.32950327, 0.32950327), (0.2481389, 0.2481389, 0.2481389)),
])

std = ActivityImageFolder("../../activity/raw_activity_small", transform_train)
std_loader = torch.utils.data.DataLoader(
    std, batch_size=4, shuffle=True, num_workers=1)


for batch_idx, (inputs, targets) in enumerate(std_loader):
	print (inputs.size(), targets.size())

#act = ActivityImageFolder(root="../../activity/raw_activity_small")


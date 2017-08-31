import argparse
import os
import commands
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from datasets import GrayImageFolder
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage

posture_tags = set(["sitting", "standing", "unknown"])
activity_tags = set(["sitting", "standing", "unknown", "walking", "up", "down", "laying", "falling"])

parser = argparse.ArgumentParser()
parser.add_argument('--type', dest='type', default='posture', choices=['posture', 'activity'], help='posture or activity dataset')
parser.add_argument('--width', dest='width', default=36, type=int, help='the width of the images')
parser.add_argument('--height', dest='height', default=28, type=int, help='the height of the images')
parser.add_argument('--root', dest='root', default='raw', help='path of archives')
parser.add_argument('--usecache', dest='usecache', default=True, type=bool, help='use dataset from cache')

args = parser.parse_args()
path = args.root
new_path = path + "_output"
usecache = args.usecache
width = args.width
height = args.height
dataset_name = args.type + "_" + str(width) + "x" + str(height) + ".data"


def get_leaf_folders():
	comm = "find " + path + " -type d -links 2"
	folders = commands.getoutput(comm)
	return folders.split('\n')

def get_type_folders(tags):
	folders_path = get_leaf_folders()
 	folders_path = list(filter(lambda folder_path: len(tags.intersection(folder_path.split('/'))) != 0, folders_path))
 	return folders_path

def compute_no_images(tags):
	folders_path = get_type_folders(posture_tags)
 	result = list(map(lambda folder_path: int(commands.getoutput("ls -1 " + folder_path + " | wc -l")), folders_path))
	return reduce(lambda x, y: x+y, result)

def prepare_posture_dataset():
	print("Preparing posture dataset...")
	folders_path = get_type_folders(posture_tags)

	for tag in posture_tags:
		if not os.path.exists(new_path + "/" + tag):
			os.makedirs(new_path + "/" + tag)

	for folder_path in folders_path:
		target_class = next(iter(posture_tags.intersection(folder_path.split('/'))))
		for img_name in os.listdir(folder_path):
			im = Image.open(folder_path+"/"+img_name)
			im_format = im.format
			im = im.convert("L")

			im_resized = im.resize((args.width, args.height), Image.ANTIALIAS)
			im_resized.save(new_path+"/"+target_class+"/"+img_name.split('.')[0]+"."+im_format.lower(), im_format)

def show(img):
    npimg = img.numpy()
    plt.imshow(npimg, interpolation='nearest')

def create_posture_dataset():
	print("Creating posture dataset")
	data = GrayImageFolder(root=new_path, transform=ToTensor())
	torch.save(data, dataset_name)
	datad =  torch.load("alabala.txt")
	#loader = torch.utils.data.DataLoader(data, batch_size=2)

	#dataiter = iter(loader)
	#images, labels = dataiter.next()
	print(datad.imgs)
	for tupl in datad.imgs:
		print(tupl)

	#images = images[:.0,:,:].reshape(images.size(0),1,images.size(2). images.size(3))
	#print(images[0,:,:,:].sum())
	#print(images[1,:,:,:].sum())
	#imgplot = plt.imshow(torchvision.utils.make_grid(images))

def prepare_activity_dataset():
	print("Preparing activity dataset...")

def create_activity_dataset():
	print("Creating activity dataset")


def main():
	if args.type=='posture':
		if usecache:
			if not os.path.exists(new_path):
				prepare_posture_dataset()
			if not os.path.isfile(dataset_name):
				create_posture_dataset()
		else:
			prepare_posture_dataset()
			create_posture_dataset()
		print("Finished")
	else:
		if not usecache:
			prepare_activity_dataset()
		else:
			print("gonna do something")

if __name__ == "__main__":
	main()
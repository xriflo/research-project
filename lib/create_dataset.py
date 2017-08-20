import argparse
import os
import commands
from random import shuffle
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from my_datasets import GrayImageFolder
import numpy as np
import pickle

posture_tags = set(["sitting", "standing", "unknown"])
activity_tags = set(["sitting", "standing", "unknown", "walking", "up", "down", "laying", "falling"])

parser = argparse.ArgumentParser()
parser.add_argument('--type', dest='type', default='posture', choices=['posture', 'activity'], help='posture or activity dataset')
parser.add_argument('--width', dest='width', default=36, type=int, help='the width of the images')
parser.add_argument('--height', dest='height', default=28, type=int, help='the height of the images')
parser.add_argument('--root', dest='root', default='raw', help='name of dataset')
parser.add_argument('--usecache', dest='usecache', default=True, type=bool, help='use dataset from cache')
parser.add_argument('--data_folder', dest='data_folder', default='../../raw', help='path of dataset')

args = parser.parse_args()
path_tr = args.data_folder + "_train"
path_ts = args.data_folder + "_test"

usecache = args.usecache
width = args.width
height = args.height

def get_leaf_folders(path):
	comm = "find " + path + " -type d -links 2"
	folders = commands.getoutput(comm)
	return folders.split('\n')

def get_type_folders(tags, path):
	folders_path = get_leaf_folders(path)
 	folders_path = list(filter(lambda folder_path: len(tags.intersection(folder_path.split('/'))) != 0, folders_path))
 	return folders_path

def compute_no_images(tags):
	folders_path = get_type_folders(posture_tags)
 	result = list(map(lambda folder_path: int(commands.getoutput("ls -1 " + folder_path + " | wc -l")), folders_path))
	return reduce(lambda x, y: x+y, result)


def prepare_posture_dataset(path):
	new_path = path + "_resized"
	mean = np.zeros(3)
	std = np.zeros(3)
	
	folders_path = get_type_folders(posture_tags, path)
	for tag in posture_tags:
		if not os.path.exists(new_path + "/" + tag):
			os.makedirs(new_path + "/" + tag)

	no_imgs = 0
	for folder_path in folders_path:
		print folder_path, "........"
		img_type = "RGB"
		target_class = next(iter(posture_tags.intersection(folder_path.split('/'))))
		for img_name in os.listdir(folder_path):
			no_imgs = no_imgs + 1
			im = Image.open(folder_path+"/"+img_name)
			im_format = im.format
			im = im.convert(img_type)
			im_resized = im.resize((args.width, args.height), Image.ANTIALIAS)
			dst = new_path+"/"+target_class+"/"+img_name
			im_resized.save(new_path+"/"+target_class+"/"+img_name.split('.')[0]+"."+im_format.lower(), im_format)
			im_numpy = np.divide(np.array(im_resized, dtype=np.float), 255)
			mean = np.add(mean, np.mean(im_numpy, axis=(0, 1)))
			std = np.add(std, np.std(im_numpy, axis=(0, 1)))
	mean = mean/no_imgs
	std = std/no_imgs
	print(mean)
	print(std)

'''
def prepare_posture_dataset():
	print("Preparing posture dataset...")
	mean = np.zeros(3)
	std = np.zeros(3)
	
	folders_path = get_type_folders(posture_tags)

	for tag in posture_tags:
		if not os.path.exists(new_path + "/" + tag):
			os.makedirs(new_path + "/" + tag)

	for folder_path in folders_path:
		img_type = "RGB"
		target_class = next(iter(posture_tags.intersection(folder_path.split('/'))))
		for img_name in os.listdir(folder_path):
			im = Image.open(folder_path+"/"+img_name)
			im_format = im.format
			im = im.convert(img_type)
			im_resized = im.resize((args.width, args.height), Image.ANTIALIAS)
			#im_resized.save(new_path+"/"+target_class+"/"+img_name.split('.')[0]+"."+im_format.lower(), im_format)
			#im_numpy = np.divide(np.array(im_resized,dtype=np.float), 255)
			#print np.mean(im_numpy, axis=(0,1))

'''
'''
def split_posture_dataset():
	mean = np.zeros(3)
	std = np.zeros(3)
	tr_samples = 0

	for tag in posture_tags:
		if not os.path.exists(new_path_tr + "/" + tag):
			os.makedirs(new_path_tr + "/" + tag)
		if not os.path.exists(new_path_ts + "/" + tag):
			os.makedirs(new_path_ts + "/" + tag)

		images = commands.getoutput("find " + new_path + "/" + tag + "/ -type f").split('\n')
		shuffle(images)
		no_images_all = len(images)
		no_images_tr = int(0.75 * no_images_all)
		tr_samples = tr_samples + no_images_tr

		for i in range(no_images_tr):
			src = images[i]
			dst = images[i].replace(new_path, new_path_tr)
			commands.getoutput("cp " + src + " " + dst)
			im_numpy = np.divide(np.array(Image.open(commands.getoutput("pwd")+"/"+src), dtype=np.float), 255)
			mean = np.add(mean, np.mean(im_numpy, axis=(0, 1)))
			std = np.add(std, np.std(im_numpy, axis=(0, 1)))

		for i in range(no_images_tr, no_images_all):
			src = images[i]
			dst = images[i].replace(new_path, new_path_ts)
			commands.getoutput("cp " + src + " " + dst)

	mean = mean/tr_samples
	std = std/tr_samples

	with open('mean_std.file', 'wb') as fp:
		pickle.dump(mean, fp)
		pickle.dump(std, fp)
	print mean, "   ", std

def show(img):
	npimg = img.numpy()
	plt.imshow(npimg, interpolation='nearest')


def create_posture_dataset():
	prepare_posture_dataset()
	split_posture_dataset()


def prepare_activity_dataset():
	print("Preparing activity dataset...")

def create_activity_dataset():
	print("Creating activity dataset")
'''





def main():
	prepare_posture_dataset(path_ts)

if __name__ == "__main__":
	main()

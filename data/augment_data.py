import os
import commands
import math
from prettytable import PrettyTable
from PIL import Image

activity_tags = set(["walking", "up", "down"])

path = "../../activity/raw"


def get_leaf_folders():
	comm = "find ./" + path + " -type d -links 2"
	folders = commands.getoutput(comm)
	return folders.split('\n')


def augment_data_flipping():
	folders = get_leaf_folders()
	for folder in folders:
		tag = activity_tags.intersection(folder.split('/'))
		if len(tag) !=0:
			tag = next(iter(tag))
			if not os.path.exists(folder+"_flipped"):
				os.makedirs(folder+"_flipped")

			for img_name in os.listdir(folder):
				im = Image.open(folder+"/"+img_name)
				im = im.transpose(Image.FLIP_LEFT_RIGHT)
				im.save(folder+"_flipped"+"/"+"flipped_"+img_name)



augment_data_flipping()
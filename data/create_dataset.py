import argparse
import os
import commands
from PIL import Image

posture_tags = set(["sitting", "standing", "unknown"])
activity_tags = set(["sitting", "standing", "unknown", "walking", "up", "down", "laying", "falling"])

parser = argparse.ArgumentParser()
parser.add_argument('--type', dest='type', default='posture', choices=['posture', 'activity'], help='posture or activity dataset')
parser.add_argument('--width', dest='width', default=36, type=int, help='the width of the images')
parser.add_argument('--height', dest='height', default=28, type=int, help='the height of the images')
parser.add_argument('--root', dest='root', default='raw_small', help='path of archives')

args = parser.parse_args()
path = args.root
new_path = path + "_output"


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
			im_resized = im.resize((args.width, args.height), Image.ANTIALIAS)
			im_resized.save(new_path+"/"+target_class+"/"+img_name.split('.')[0]+"."+im.format.lower(), im.format)


def prepare_activity_dataset():
	print("Preparing activity dataset...")


def main():
	if args.type=='posture':
		prepare_posture_dataset()
	else:
		prepare_activity_dataset()

if __name__ == "__main__":
    main()
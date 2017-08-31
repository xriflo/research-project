from random import randint
from PIL import Image
import os
import commands

NO_SEQ = 16
WIDTH = 48
HEIGHT = 42
#rules for creating dataset
activity_tags = set(["sitting", "standing", "walking", "up", "down"])
path = "../../activity/raw"
new_path = path + "_activity"



rules = {}
#[[a,b,c]] - take c combinations from a sequence with b <= nopics < c
rules['standing'] = [
					[16,float('inf'),1]
					]
rules['sitting'] = [
					[16,100,1], 
					[100,float('inf'),3]
					]
rules['up'] = [
				[16,30,1], 
				[30,50,2], 
				[50,100,2], 
				[100,float('inf'),3]
					]
rules['down'] = [
				[16,30,1], 
				[30,float('inf'),2]
				]
rules['walking'] = [
					[16,50,1], 
					[50,100,2], 
					[100,300,3], 
					[300,1000,8],  
					[1000,float('inf'),70]
					]



def find_no_combinations(tag, no_pics):
	specific_rules = rules[tag]
	no_combinations = None
	for rule in specific_rules:
		if no_pics >= rule[0] and no_pics < rule[1]:
			no_combinations = rule[2]
	return no_combinations

def find_all_datasets():
	folders = get_leaf_folders()
	datasets = set([])
	for folder in folders:
		datasets.add(folder.split('/')[5])
	return datasets

def create_dataset_structure():
	datasets = find_all_datasets()
	for tag in activity_tags:
		for dataset in datasets:
			path_to_create = new_path + "/" + tag + "/" + dataset
			if not os.path.exists(path_to_create):
				os.makedirs(path_to_create)


def find_balanced_combinations(no_pics, how_many):
	combinations = []

	step = float(no_pics)/NO_SEQ
	intervals = []
	start = 0
	end = start + step
	for i in range(1, NO_SEQ):
		start_to_int = int(start)
		end_to_int = int(end)
		intervals.append([start_to_int, end_to_int])
		start = start + step
		end = end + step
	intervals.append([end_to_int, int(no_pics)])

	for i in range(how_many):
		combination = [0]
		for interval in intervals[1:NO_SEQ-1]:
			combination.append(randint(interval[0], interval[1]-1))
		combination.append(intervals[-1][-1]-1)
		combinations.append(combination)
	return combinations


def get_leaf_folders():
	comm = "find ./" + path + " -type d -links 2"
	folders = commands.getoutput(comm)
	return folders.split('\n')


def create_dataset():
	create_dataset_structure()
	seq = 0
	folders = get_leaf_folders()
	for folder in folders:
		tag = activity_tags.intersection(folder.split('/'))
		if len(tag)!=0:
			comm = "ls -1 " + folder + " | wc -l"
			no_pics = int(commands.getoutput(comm))
			if no_pics >= NO_SEQ:
				dataset_name = folder.split('/')[5]
				tag = next(iter(tag))
				how_many = find_no_combinations(tag, no_pics)
				#print "no_pics=", no_pics, "       how_many=", how_many, "      tag=", tag
				combinations = find_balanced_combinations(no_pics, how_many)
				
				images_name = os.listdir(folder)
				images_name.sort()
				for combination in combinations:
					seq = seq + 1 
					img_names = map(lambda i: images_name[i], combination)
					for img_name in img_names:
						old_path_pic = folder + "/" + img_name
						im = Image.open(old_path_pic)
						im_format = im.format
						im = im.convert("L")
						im_resized = im.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
						new_base_path_pic = new_path + "/" + tag + "/" + dataset_name + "/" + str(seq) + "/"
						new_path_pic = new_base_path_pic + img_name.split('.')[0] + "." + im_format.lower()
						if not os.path.exists(new_base_path_pic):
							os.makedirs(new_base_path_pic)
						im_resized.save(new_path_pic, im_format)


#print(find_balanced_combinations(93, 2))
create_dataset()
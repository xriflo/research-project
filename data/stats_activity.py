import os
import commands
import math
from prettytable import PrettyTable

activity_tags = set(["sitting", "standing", "unknown", "walking", "up", "down", "falling"])

path = "../../activity/raw"


def create_dicts():
	no_pictures_per_class = {}
	no_seq_per_class = {}
	mean_per_class = {}
	std_per_class = {}
	var_per_class = {}
	min_per_class = {}
	max_per_class = {}
	for tag in activity_tags:
		no_pictures_per_class[tag]=0
		no_seq_per_class[tag]=0
		mean_per_class[tag]=0
		std_per_class[tag]=0
		var_per_class[tag]=0
		min_per_class[tag]=float('inf')
		max_per_class[tag]=-float('inf')
	return no_pictures_per_class, no_seq_per_class, mean_per_class, std_per_class, var_per_class, min_per_class, max_per_class

def get_leaf_folders():
	comm = "find ./" + path + " -type d -links 2"
	folders = commands.getoutput(comm)
	return folders.split('\n')

def pretty_print(dictt, name):
	print "-----------------------------------------------------------"
	print name, " -> "
	for k, v in dictt.items():
		print k,":",v," "


def get_stats():
	no_pictures_per_class, no_seq_per_class, mean_per_class, std_per_class, var_per_class, min_per_class, max_per_class = create_dicts()
	folders = get_leaf_folders()

	for folder in folders:
		tag = activity_tags.intersection(folder.split('/'))
		if len(tag) !=0:
			tag = next(iter(tag))
			comm = "ls -1 " + folder + " | wc -l"
			no_pics = int(commands.getoutput(comm))

			no_pictures_per_class[tag] = no_pictures_per_class[tag] + no_pics
			no_seq_per_class[tag] = no_seq_per_class[tag] + 1
			mean_per_class[tag] = mean_per_class[tag] + no_pics

			if no_pics <= 10 and tag=='sitting':
				print folder

			if no_pics < min_per_class[tag]:
				min_per_class[tag] = no_pics

			if no_pics > max_per_class[tag]:
				max_per_class[tag] = no_pics


	for tag in activity_tags:
		mean_per_class[tag] = mean_per_class[tag] / no_seq_per_class[tag]

	for folder in folders:
		tag = activity_tags.intersection(folder.split('/'))
		if len(tag)!=0:
			tag = next(iter(tag))
			comm = "ls -1 " + folder + " | wc -l"
			no_pics = int(commands.getoutput(comm))
			var_per_class[tag] = var_per_class[tag] + (no_pics - mean_per_class[tag])**2

	for tag in activity_tags:
		var_per_class[tag] = var_per_class[tag] / no_seq_per_class[tag]
		std_per_class[tag] = math.sqrt(var_per_class[tag])


	t = PrettyTable(['class', 'no of pictures per class', 'no of seq per class', 'min', 'max', 'mean', 'std', 'var'])
	for tag in activity_tags:
		row = [tag, no_pictures_per_class[tag], no_seq_per_class[tag], min_per_class[tag], max_per_class[tag], mean_per_class[tag], std_per_class[tag], var_per_class[tag]]
		t.add_row(row)

	print(t)


'''
			pics[tag] = pics[tag] + 1
			comm = "ls -1 ./ " + folder + " | wc -l"
			no_pics = int(commands.getoutput(comm))
			pics[tag] = pics[tag] + no_pics
			if tag == 'unknown':
				print no_pics, ":", folder
	print pics
'''
get_stats()

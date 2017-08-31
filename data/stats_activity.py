import os
import commands
import math
from prettytable import PrettyTable

activity_tags = set(["sitting", "standing", "walking", "up", "down"])

path = "../../activity/raw"


def create_dicts():
	no_pictures_per_class = {}
	no_seq_per_class = {}
	mean_per_class = {}
	std_per_class = {}
	var_per_class = {}
	min_per_class = {}
	max_per_class = {}
	no_seq_16_100 = {}
	no_seq_100_300 = {}
	no_seq_300_1000 = {}
	no_seq_1000_inf = {}
	for tag in activity_tags:
		no_pictures_per_class[tag]=0
		no_seq_per_class[tag]=0
		mean_per_class[tag]=0
		std_per_class[tag]=0
		var_per_class[tag]=0
		min_per_class[tag]=float('inf')
		max_per_class[tag]=-float('inf')
		no_seq_16_100[tag]=0
		no_seq_100_300[tag]=0
		no_seq_300_1000[tag]=0
		no_seq_1000_inf[tag]=0
	return no_pictures_per_class, no_seq_per_class, mean_per_class, std_per_class, var_per_class, min_per_class, max_per_class, no_seq_16_100, no_seq_100_300, no_seq_300_1000, no_seq_1000_inf

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
	no_pictures_per_class, no_seq_per_class, mean_per_class, std_per_class, var_per_class, min_per_class, max_per_class, no_seq_16_100, no_seq_100_300, no_seq_300_1000, no_seq_1000_inf = create_dicts()
	folders = get_leaf_folders()

	interval0 = 16
	intervalA = 30
	intervalB = 50
	intervalC = 100

	for folder in folders:
		tag = activity_tags.intersection(folder.split('/'))
		if len(tag) !=0:
			tag = next(iter(tag))
			comm = "ls -1 " + folder + " | wc -l"
			no_pics = int(commands.getoutput(comm))
			if no_pics >= interval0:
				no_pictures_per_class[tag] = no_pictures_per_class[tag] + no_pics
				no_seq_per_class[tag] = no_seq_per_class[tag] + 1
				mean_per_class[tag] = mean_per_class[tag] + no_pics

				if no_pics < intervalA:
					no_seq_16_100[tag] = no_seq_16_100[tag] + 1
				elif no_pics >= intervalA and no_pics < intervalB:
					no_seq_100_300[tag] = no_seq_100_300[tag] + 1
				elif no_pics >= intervalB and no_pics < intervalC:
					no_seq_300_1000[tag] = no_seq_300_1000[tag] + 1
				else:
					no_seq_1000_inf[tag] = no_seq_1000_inf[tag] + 1


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
			if no_pics >= 16:
				no_pics = int(commands.getoutput(comm))
				var_per_class[tag] = var_per_class[tag] + (no_pics - mean_per_class[tag])**2

	for tag in activity_tags:
		var_per_class[tag] = var_per_class[tag] / no_seq_per_class[tag]
		std_per_class[tag] = int(math.sqrt(var_per_class[tag]))


	t = PrettyTable(['class', 'no of pictures per class', 'no of seq per class', 'min', 'max', 'mean', 'std', 'var', str(interval0)+'-'+str(intervalA),  str(intervalA)+'-'+str(intervalB),  str(intervalB)+'-'+str(intervalC),  str(intervalC)+' - inf'])
	for tag in activity_tags:
		if tag=='sitting':
			row = [tag, no_pictures_per_class[tag], no_seq_per_class[tag], min_per_class[tag], max_per_class[tag], mean_per_class[tag], std_per_class[tag], var_per_class[tag], no_seq_16_100[tag], no_seq_100_300[tag], no_seq_300_1000[tag], no_seq_1000_inf[tag]]
			t.add_row(row)

	print(t)


get_stats()

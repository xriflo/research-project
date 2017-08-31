from random import randint
import os
import commands

NO_SEQ = 16
#rules for creating dataset
activity_tags = set(["sitting", "standing", "walking", "up", "down"])
path = "../../activity/raw_very_small"
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
					[16,30,1], 
					[30,100,2], 
					[100,300,3], 
					[300,100,6], 
					[1000,float('inf'),62]
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
				combinations = find_balanced_combinations(no_pics, how_many)
				images_name = os.listdir(folder)
				images_name.sort()

				for combination in combinations:
					seq = seq + 1 
					new_activity = map(lambda i: images_name[i], combination)
					print new_activity

		break

print(find_all_datasets())
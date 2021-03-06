from random import randint
from PIL import Image
import os
import commands

NO_SEQ = 16
MAX_STEP = 3
WIDTH = 48
HEIGHT = 42
#rules for creating dataset
activity_tags = set(["sitting", "standing", "walking", "up", "down"])
path = path = "../../archives_mihai/train"
new_path = path + "_activity"


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


def find_balanced_combinations_twist(tag, no_pics, how_many):
	if tag!='walking' or (tag=='walking' and no_pics < 50):
		return find_balanced_combinations(no_pics, how_many)
	else:
		combinations = []
		for i in range(how_many):
			combination = []
			start = randint(0, no_pics - MAX_STEP*NO_SEQ)
			combination.append(start)
			for j in range(1, NO_SEQ):
				start = start + randint(1, MAX_STEP)
				combination.append(start)
			combinations.append(combination)
		return combinations


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

def create_dicts(tags):
	no_pictures_per_class = {}
	no_seq_per_class = {}
	mean_per_class = {}
	std_per_class = {}
	var_per_class = {}
	min_per_class = {}
	max_per_class = {}
	
	no_seq_0_A = {}
	no_seq_A_B = {}
	no_seq_B_C = {}
	no_seq_C_D = {}
	no_seq_D_E = {}
	no_seq_E_inf = {}

	for tag in tags:
		no_pictures_per_class[tag]=0
		no_seq_per_class[tag]=0
		mean_per_class[tag]=0
		std_per_class[tag]=0
		var_per_class[tag]=0
		min_per_class[tag]=float('inf')
		max_per_class[tag]=-float('inf')
		no_seq_0_A[tag]=0
		no_seq_A_B[tag]=0
		no_seq_B_C[tag]=0
		no_seq_C_D[tag]=0
		no_seq_D_E[tag]=0
		no_seq_E_inf[tag]=0
	return no_pictures_per_class, no_seq_per_class, mean_per_class, std_per_class, var_per_class, min_per_class, max_per_class, no_seq_0_A, no_seq_A_B, no_seq_B_C, no_seq_C_D, no_seq_D_E, no_seq_E_inf


def create_dataset():
	def get_stats():
	folders = get_leaf_folders()

	interval0 = 16
	intervalA = 30
	intervalB = 50
	intervalC = 100
	intervalD = 300
	intervalE = 1000

	tags_set = set([])

	for folder in folders:
		tag = str(folder[-4:])
		tags_set.add(tag)

	no_pictures_per_class, no_seq_per_class, mean_per_class, std_per_class, var_per_class, min_per_class, max_per_class, no_seq_0_A, no_seq_A_B, no_seq_B_C, no_seq_C_D, no_seq_D_E, no_seq_E_inf = create_dicts(tags_set)

	#compute no of sequences
	for folder in folders:
		tag = str(folder[-4:])

		if len(tag) !=0:
			comm = "ls -1 " + folder + " | wc -l"
			no_pics = int(commands.getoutput(comm))

			no_pictures_per_class[tag] = no_pictures_per_class[tag] + no_pics
			no_seq_per_class[tag] = no_seq_per_class[tag] + 1
			mean_per_class[tag] = mean_per_class[tag] + no_pics
			if no_pics==0:
				print folder
			if no_pics < intervalA:
				no_seq_0_A[tag] = no_seq_0_A[tag] + 1
			elif no_pics >= intervalA and no_pics < intervalB:
				no_seq_A_B[tag] = no_seq_A_B[tag] + 1
			elif no_pics >= intervalB and no_pics < intervalC:
				no_seq_B_C[tag] = no_seq_B_C[tag] + 1
			elif no_pics >= intervalC and no_pics < intervalD:
				no_seq_C_D[tag] = no_seq_C_D[tag] + 1
			elif no_pics >= intervalD and no_pics < intervalE:
				no_seq_D_E[tag] = no_seq_D_E[tag] + 1
			else:
				no_seq_E_inf[tag] = no_seq_E_inf[tag] + 1


			if no_pics < min_per_class[tag]:
				min_per_class[tag] = no_pics

			if no_pics > max_per_class[tag]:
				max_per_class[tag] = no_pics
	sum_sequences = {}
	for tag in tags_set:
		sum_sequences[tag] = no_seq_A_B[tag] + no_seq_B_C[tag] + no_seq_C_D[tag]


	seq = 0
	folders = get_leaf_folders()
	for folder in folders:
		tag = str(folder[-4:])

		if len(tag)!=0:
			comm = "ls -1 " + folder + " | wc -l"
			no_pics = int(commands.getoutput(comm))

			if no_pics >= NO_SEQ:
				#compute how many combination
				if no_pics < intervalB:
					how_many = int((no_seq_A_B[tag]*100.0)/float(sum_sequences[tag]))
					combinations = find_balanced_combinations(no_pics, how_many)
				elif no_pics >= intervalB and no_pics < intervalC:
					how_many = int((no_seq_B_C[tag]*100.0)/float(sum_sequences[tag]))
					combinations = find_balanced_combinations(no_pics, how_many)
				else:
					how_many = int((no_seq_C_D[tag]*100.0)/float(sum_sequences[tag]))
					combinations = find_balanced_combinations(no_pics, how_many)
				
				images_name = os.listdir(folder)
				images_name.sort()
				print combinations
				'''
				for combination in combinations:
					seq = seq + 1 
					img_names = map(lambda i: images_name[i], combination)
					for img_name in img_names:
						old_path_pic = folder + "/" + img_name
						im = Image.open(old_path_pic)
						im_format = im.format
						im = im.convert("L")
						im_resized = im.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
						new_base_path_pic = new_path + "/" + tag + "/" + dataset_name + "/" + episode + "/" + str(seq) + "/"
						new_path_pic = new_base_path_pic + img_name.split('.')[0] + "." + im_format.lower()
						if not os.path.exists(new_base_path_pic):
							os.makedirs(new_base_path_pic)
						im_resized.save(new_path_pic, im_format)
				'''


#print(find_balanced_combinations(93, 2))
create_dataset()
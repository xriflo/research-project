from random import randint
from PIL import Image
import os
import commands
from prettytable import PrettyTable
import math

NO_SEQ = 16
MAX_STEP = 3
WIDTH = 48
HEIGHT = 42
#rules for creating dataset
activity_tags = set(["sitting", "standing", "walking", "up", "down"])
path = path = "../../archives_mihai/test"
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
	if how_many == 0:
		return combinations
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
	no_seq_C_inf = {}


	for tag in tags:
		no_pictures_per_class[tag]=0
		no_seq_per_class[tag]=0
		mean_per_class[tag]=0
		std_per_class[tag]=0
		var_per_class[tag]=0
		min_per_class[tag]=float('inf')
		max_per_class[tag]=-float('inf')

		no_seq_A_B[tag]=0
		no_seq_B_C[tag]=0
		no_seq_C_inf[tag]=0

	return no_pictures_per_class, no_seq_per_class, mean_per_class, std_per_class, var_per_class, min_per_class, max_per_class, no_seq_A_B, no_seq_B_C, no_seq_C_inf


def create_dataset():

	folders = get_leaf_folders()

	intervalA = 30
	intervalB = 50
	intervalC = 100

	tags_set = set([])

	for folder in folders:
		tag = str(folder[-4:])
		tags_set.add(tag)

	no_pictures_per_class, no_seq_per_class, mean_per_class, std_per_class, var_per_class, min_per_class, max_per_class, no_seq_A_B, no_seq_B_C, no_seq_C_inf = create_dicts(tags_set)

	#compute no of sequences
	for folder in folders:
		tag = str(folder[-4:])
		comm = "ls -1 " + folder + " | wc -l"
		no_pics = int(commands.getoutput(comm))

		if len(tag) !=0 and no_pics >= intervalA:
			
			no_pictures_per_class[tag] = no_pictures_per_class[tag] + no_pics
			no_seq_per_class[tag] = no_seq_per_class[tag] + 1
			mean_per_class[tag] = mean_per_class[tag] + no_pics

			if no_pics >= intervalA and no_pics < intervalB:
				no_seq_A_B[tag] = no_seq_A_B[tag] + 1
			elif no_pics >= intervalB and no_pics < intervalC:
				no_seq_B_C[tag] = no_seq_B_C[tag] + 1
			else:
				no_seq_C_inf[tag] = no_seq_C_inf[tag] + 1


			if no_pics < min_per_class[tag]:
				min_per_class[tag] = no_pics

			if no_pics > max_per_class[tag]:
				max_per_class[tag] = no_pics




	sum_sequences = {}
	for tag in tags_set:
		sum_sequences[tag] = no_seq_A_B[tag] + no_seq_B_C[tag] + no_seq_C_inf[tag]
	'''
	header = ['class', 'no of pictures per class', 'no of seq per class', 'min', 'max',
		'seq: '+str(intervalA)+'-'+str(intervalB),  
		'how_many: '+str(intervalA)+'-'+str(intervalB), 
		'seq: '+str(intervalB)+'-'+str(intervalC),
		'how_many: '+str(intervalB)+'-'+str(intervalC),
		'seq: '+str(intervalC)+'-inf',
		'how_many: '+str(intervalC)+'-inf'
		]
	#print ("header size: ", len(header))
	t = PrettyTable(header)

	for tag in tags_set:
		row = [tag, no_pictures_per_class[tag], no_seq_per_class[tag], min_per_class[tag], max_per_class[tag],
				no_seq_A_B[tag],
				int(no_seq_A_B[tag]*2000.0/sum_sequences[tag]),
				no_seq_B_C[tag],
				int(no_seq_B_C[tag]*2000.0/sum_sequences[tag]),
				no_seq_C_inf[tag],
				int(no_seq_C_inf[tag]*2000.0/sum_sequences[tag])]
		#print ("row size: ", len(row))
		t.add_row(row)
	print(t)
	'''


	seq = 0
	folders = get_leaf_folders()

	for folder in folders:
		if seq%1000==0:
			print seq
		tag = str(folder[-4:])

		if len(tag)!=0:
			comm = "ls -1 " + folder + " | wc -l"
			no_pics = int(commands.getoutput(comm))

			how_many = 0
			repeat_each = int(math.ceil(250.0/sum_sequences[tag]))
			'''
			if no_pics >= intervalA and no_pics < intervalB:
				how_many = int(no_seq_A_B[tag]*2000.0/sum_sequences[tag])
			elif no_pics >= intervalB and no_pics < intervalC:
				how_many = int(no_seq_B_C[tag]*2000.0/sum_sequences[tag])
			else:
				how_many = int(no_seq_C_inf[tag]*2000.0/sum_sequences[tag])
			'''
			combinations = find_balanced_combinations(no_pics, repeat_each)


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
					new_base_path_pic = new_path + "/" + tag +  "/" + str(seq) + "/"
					new_path_pic = new_base_path_pic + img_name.split('.')[0] + "." + im_format.lower()

					if not os.path.exists(new_base_path_pic):
						os.makedirs(new_base_path_pic)
					im_resized.save(new_path_pic, im_format)



create_dataset()
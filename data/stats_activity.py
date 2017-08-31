import os
import commands
import math
from prettytable import PrettyTable

activity_tags = set(["sitting", "standing", "walking", "up", "down"])

path = "../../raw2"


def create_dicts():
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

	for tag in activity_tags:
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
	no_pictures_per_class, no_seq_per_class, mean_per_class, std_per_class, var_per_class, min_per_class, max_per_class, no_seq_0_A, no_seq_A_B, no_seq_B_C, no_seq_C_D, no_seq_D_E, no_seq_E_inf = create_dicts()
	folders = get_leaf_folders()

	interval0 = 16
	intervalA = 30
	intervalB = 50
	intervalC = 100
	intervalD = 300
	intervalE = 1000

	for folder in folders:
		tag = activity_tags.intersection(folder.split('/'))
		if len(tag) !=0:
			tag = next(iter(tag))
			comm = "ls -1 " + folder + " | wc -l"
			no_pics = int(commands.getoutput(comm))

			no_pictures_per_class[tag] = no_pictures_per_class[tag] + no_pics
			no_seq_per_class[tag] = no_seq_per_class[tag] + 1
			mean_per_class[tag] = mean_per_class[tag] + no_pics

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
		std_per_class[tag] = int(math.sqrt(var_per_class[tag]))


		t = PrettyTable(['class', 'no of pictures per class', 'no of seq per class', 'min', 'max', 'mean', 'std', 'var', 
		str(interval0)+'-'+str(intervalA),  
		str(intervalA)+'-'+str(intervalB),  
		str(intervalB)+'-'+str(intervalC),
		str(intervalC)+'-'+str(intervalD),
		str(intervalD)+'-'+str(intervalE),  
		str(intervalE)+' - inf'])

	for tag in activity_tags:
		row = [tag, no_pictures_per_class[tag], no_seq_per_class[tag], min_per_class[tag], max_per_class[tag], mean_per_class[tag], std_per_class[tag], var_per_class[tag], no_seq_0_A[tag], no_seq_A_B[tag], no_seq_B_C[tag], no_seq_C_D[tag], no_seq_D_E[tag], no_seq_E_inf[tag]]
		t.add_row(row)

	print(t)


get_stats()

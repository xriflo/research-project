import os
import commands


posture_tags = set(["sitting", "standing", "unknown"])

path = "../../raw_test"


def create_dicts():
	all_pics = {}
	for tag in posture_tags:
		all_pics[tag]=0
	return all_pics

pics = create_dicts()

def get_leaf_folders():
	comm = "find ./" + path + " -type d -links 2"
	folders = commands.getoutput(comm)
	return folders.split('\n')



def get_stats():
	folders = get_leaf_folders()
	for folder in folders:
		tag = posture_tags.intersection(folder.split('/'))
		if len(tag) !=0:
			tag = next(iter(tag))
			pics[tag] = pics[tag] + 1
			comm = "ls -1 ./ " + folder + " | wc -l"
			no_pics = int(commands.getoutput(comm))
			pics[tag] = pics[tag] + no_pics
			if tag == 'unknown':
				print no_pics, ":", folder
	print pics

get_stats()

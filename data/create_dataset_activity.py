


#rules for creating dataset
activity_tags = set(["sitting", "standing", "walking", "up", "down"])

rules = {}
#[[a,b,c]] - take c combinations from a sequence with b <= nopics < c
rules['standing'] = [[16,float('inf'),1]]
rules['sitting'] = [[16,100,1], [100,float('inf'),3]]
rules['up'] = [[16,30,1], [30,50,2], [50,100,2], [100,float('inf'),3]]
rules['down'] = [[16,30,1], [30,float('inf'),2]]
rules['walking'] = [[16,30,1], [30,100,2], [100,300,3], [300,100,6], [1000,float('inf'),62]]



def find_no_combinations(tag, no_pics):
	specific_rules = rules[tag]
	no_combinations = None
	for rule in specific_rules:
		if no_pics >= rule[0] and no_pics < rule[1]:
			no_combinations = rule[2]
	return no_combinations




def get_leaf_folders():
	comm = "find ./" + path + " -type d -links 2"
	folders = commands.getoutput(comm)
	return folders.split('\n')


for folder in folders:
	tag = activity_tags.intersection(folder.split('/'))
	if len(tag) !=0:
		tag = next(iter(tag))
		comm = "ls -1 " + folder + " | wc -l"
		no_pics = int(commands.getoutput(comm))
		no_combinations = find_no_combinations(tag, no_pics)
		pisici
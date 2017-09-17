
with open('champId_champName.psv') as f:
	id_to_name = dict()
	name_to_id = dict()
	for line in f:
		str_list = line.split(' ', 2)
		champId = int(str_list[0])
		champName = str_list[1][ :-1]
		id_to_name[champId] = champName
		name_to_id[champName] = champId

if __name__ == '__main__':
	tmp = '.'
	print('Empty input for exit')
	while not tmp == '':
		tmp = input("id: ")
		try:
			mainChampId = int(tmp)
			print(id_to_name[mainChampId], '\n')
		except:
			pass



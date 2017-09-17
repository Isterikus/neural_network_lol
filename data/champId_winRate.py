
id_to_wr = dict()
wr_to_id = dict()

with open('champId_winRate.psv') as f:
	for line in f:
		str_list = line.split(' ', 2)
		champId = int(str_list[0])
		winRate = float(str_list[1].replace('\n', ''))
		id_to_wr[champId] = winRate
		wr_to_id[winRate] = champId

print(id_to_wr)
def getWr(champId):
	return id_to_wr[champId]

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



import requests
import json
import collections
import statistics

data_path = "../../data/lol/"

r = requests.get(r"http://api.champion.gg/v2/champions?champData=winRate&limit=200&api_key=fdbf5df90189c7cb9d5849731fddd7be")
champs = json.loads(r.text)

with open(data_path + "champId_winRate.psv", 'w') as f:
	arr = dict()
	for champ in champs:
		champ_id = champ['championId']
		if (not champ_id in arr):
			arr[champ_id] = [champ['winRate']]
		else:
			arr[champ_id].append(champ['winRate'])

	print(arr[13])
	od = collections.OrderedDict(sorted(arr.items()))
	#print(od)
	for champId, champWinRates in od.items():
		f.write(str(champId) + ' ' + str(statistics.median(champWinRates)) + '\n')

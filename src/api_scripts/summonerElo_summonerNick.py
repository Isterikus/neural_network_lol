import re
import requests

data_path = "../../data/lol/"
parse_servers = ["EUW", "NA", "KR"]
site = "https://www.leagueofgraphs.com/rankings/summoners"
regular_name = "<span class=\"name\">(.+)</span>"
regular_server = "<i>([A-Z]+)</i>"

with open(data_path + "summonerNick_serverId.psv", "w") as f:
	for i in range(20):
		if i == 0:
			test = requests.get(site)
		else:
			test = requests.get(site + "/page-" + str(i + 1))
		names = re.findall(regular_name, test.content)
		servers = re.findall(regular_server, test.content)
		for j in range(len(names)):
			if servers[j] in parse_servers:
				f.write('"' + names[j] + '" "' + servers[j] + '"\n')
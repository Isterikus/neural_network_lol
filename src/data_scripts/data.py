'''
champ   - champion
id      - unique id for champions
name    - champion name
winRate - champion middle (wins / looses)
'''
import csv
from functools import partial

data_path = "../../data/lol/"
rd = partial(csv.reader, delimiter=' ')
wr = partial(csv.writer, delimiter=' ')

with open(data_path + 'champId_winRate.psv') as f:
	dt = rd(f)
	ret = [{'champId': champId, 'winRate': winRate} for champId,winRate in dt]
# print(ret)
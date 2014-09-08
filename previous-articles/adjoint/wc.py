import requests
import json
import pyxdameraulevenshtein

matches = json.loads(requests.get('http://worldcup.kimonolabs.com/api/matches?sort=startTime&fields=homeScore,awayScore,startTime,awayTeamId,homeTeamId,id&apikey=72519cb45986ce5ffd15020a5e4b1a70').content)
print matches
gabriel_teams = list(set('Brazil,Mexico,Spain,Chile,Colombia,Cote Divoire,Uruguay,England,Switzerland,France,Argentina,Iran,Germany,Ghana,Belgium,Russia,Brazil,Cameroon,Spain,Australia,Colombia,Japan,Uruguay,Italy,Switzerland,Honduras,Argentina,Nigeria,Germany,USA,Belgium,Korea Republic,Cameroon,Croatia,Australia,Netherlands,Japan,Greece,Italy,Costa Rica,Honduras,Ecuador,Nigeria,Bosnia-Herzegovina,USA,Portugal,Korea Republic,Algeria'.split(',')))
teams = json.loads(requests.get('http://worldcup.kimonolabs.com/api/teams?apikey=72519cb45986ce5ffd15020a5e4b1a70').content)
names = list(set([team['name'].encode('utf-8') for team in teams]))
print names
matches = [teams[min([(i,pyxdameraulevenshtein.normalized_damerau_levenshtein_distance(gteam, team)) for i, team in enumerate(names)], key = lambda x: x[1])[0]]['id'] for gteam in gabriel_teams]
for a,b in zip(gabriel_teams, matches):
	try:
		print a,unicode(b)
	except:
		pass

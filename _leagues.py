import pandas as pd
df = pd.read_csv('oe_2026.csv', usecols=['league','gameid','participantid','teamname','date'], low_memory=False)
team = df[df['participantid'].isin([100,200])]
# Ligas con # de juegos
g = team.groupby('league')['gameid'].nunique().sort_values(ascending=False)
print('Ligas con >=50 juegos en 2026:')
for lg, n in g[g>=50].items():
    print(f'  {lg:8} {n:4} juegos')
print('\n--- LCK: juegos por equipo ---')
lck = team[team['league']=='LCK']
per = lck.groupby('teamname')['gameid'].nunique().sort_values(ascending=False)
print(per.to_string())
print(f'\nLCK: {lck.gameid.nunique()} juegos unicos | fecha mas reciente: {lck.date.max()[:10]}')

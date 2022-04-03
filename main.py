from agents1.colorblind import Colorblind
from agents1.strong import Strong
from bw4t.BW4TWorld import  BW4TWorld, createwordsettings
from bw4t.statistics import Statistics
from agents1.BW4TBaselineAgent import BaseLineAgent
from agents1.BW4THuman import Human
from agents1.Liar import Liar
from agents1.Lazy import Lazy
import os
import csv 

def getSuccesRate():
  # 
  path = "world_1"
  succes = 0
  failed = 0
  dir_list = os.listdir(path)
  ticks = 0
  for csv1 in dir_list:
    data = []
    with open('world_1/'+ csv1) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      for row in csv_reader:
        if row:  # avoid blank lines
          # print(row[0].split(';')[0])
          data.append([row[0].split(';')[0], row[0].split(';')[-1]])
    if data[-1][0] == "True": succes +=1 
    else: failed +=1
    ticks += int(data[-1][-1])
  print((succes/(succes+failed))*100, ticks/(succes+failed))

"""
This runs a single session. You have to log in on localhost:3000 and 
press the start button in god mode to start the session.
"""
for i in range(30): 
  if __name__ == "__main__":
      agents = [
        {'name': 'Color1', 'botclass': Colorblind, 'settings': {}},
        {'name': 'Color2', 'botclass': Colorblind, 'settings': {}},
          {'name':'Liar1', 'botclass':Liar, 'settings':{}},
          {'name':'Liar2', 'botclass':Liar, 'settings':{}},
          {'name': 'Lazy1', 'botclass': Lazy, 'settings': {}},
          {'name': 'Lazy2', 'botclass': Lazy, 'settings': {}},
          {'name': 'Strong1', 'botclass': Strong, 'settings': {}},
          {'name': 'Strong2', 'botclass': Strong, 'settings': {}},
          
          ]
      
      print("Started world...")
      wordsetttings = createwordsettings()
      wordsetttings['tick_duration'] = 0.00000001 #set lower for faster speed, default seems 0.1
      world=BW4TWorld(agents, worldsettings=wordsetttings).run()
      print("DONE!", i)
      print(Statistics(world.getLogger().getFileName()))
      getSuccesRate()
      
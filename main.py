from agents1.colorblind import Colorblind
from bw4t.BW4TWorld import  BW4TWorld, createwordsettings
from bw4t.statistics import Statistics
from agents1.BW4TBaselineAgent import BaseLineAgent
from agents1.BW4THuman import Human
from agents1.Liar import Liar
from agents1.Lazy import Lazy


"""
This runs a single session. You have to log in on localhost:3000 and 
press the start button in god mode to start the session.
"""
for i in range(1,100): 
  if __name__ == "__main__":
      agents = [
        {'name': 'Color', 'botclass': Colorblind, 'settings': {}},
          {'name':'Liar', 'botclass':Liar, 'settings':{}},
          {'name':'human', 'botclass':Human, 'settings':{}},
          {'name': 'Lazy', 'botclass': Lazy, 'settings': {}}
          ]
       
      print("Started world...")
      wordsetttings = createwordsettings()
      wordsetttings['tick_duration'] = 0.00000001 #set lower for faster speed, default seems 0.1
      world=BW4TWorld(agents, worldsettings=wordsetttings).run()
      print("DONE!")
      print(Statistics(world.getLogger().getFileName()))
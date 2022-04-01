from bw4t.BW4TWorld import DEFAULT_WORLDSETTINGS, BW4TWorld
from bw4t.statistics import Statistics
from agents1.BW4TBaselineAgent import BaseLineAgent
from agents1.BW4THuman import Human
from agents1.Liar import Liar
from agents1.colorblind import Colorblind


"""
This runs a single session. You have to log in on localhost:3000 and 
press the start button in god mode to start the session.
"""

if __name__ == "__main__":
    agents = [
        {'name':'Color', 'botclass':Colorblind, 'settings':{}},
        {'name':'Liar', 'botclass':Liar, 'settings':{}},
        {'name':'human', 'botclass':Human, 'settings':{}}
        ]

    print("Started world...")
    wordsetttings = DEFAULT_WORLDSETTINGS
    wordsetttings['tick_duration'] = 0.001
    world=BW4TWorld(agents, worldsettings=wordsetttings).run()
    print("DONE!")
    print(Statistics(world.getLogger().getFileName()))

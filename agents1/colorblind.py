from typing import final, List, Dict, Final
import enum, random
from bw4t.BW4TBrain import BW4TBrain
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject
from matrx.messages.message import Message

class BaseLineAgent(BW4TBrain):
    
    def __init__(self, settings:Dict[str,object]):
        super().__init__(settings)
        
    def initialize(self):
        super().initialize()
        
    def filter_bw4t_observations(self, state):
        return state
    
    def decide_on_bw4t_action(self, state:State):
            return None

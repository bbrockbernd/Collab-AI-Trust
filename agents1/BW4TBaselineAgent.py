import csv
import os
from collections import Counter
from typing import final, List, Dict, Final
import enum, random
from bw4t.BW4TBrain import BW4TBrain
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject
from matrx.messages.message import Message

class Phase(enum.Enum):
    PLAN_PATH_TO_CLOSED_DOOR=1,
    FOLLOW_PATH_TO_CLOSED_DOOR=2,
    OPEN_DOOR=3

class MessageType(enum.Enum):
    MOVING = 1
    OPENING = 2
    SEARCHING = 3
    FOUND = 4
    PICKING_UP = 5
    DROPPED = 6
    INVALID = 7

class BaseLineAgent(BW4TBrain):

    def __init__(self, settings:Dict[str,object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []
        self._log = {}

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, 
            action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):
        return state

    def decide_on_bw4t_action(self, state:State):
        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member!=agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
        # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBelief(agent_name, self._teamMembers, receivedMessages)
        
        while True:
            if Phase.PLAN_PATH_TO_CLOSED_DOOR==self._phase:
                self._navigator.reset_full()
                closedDoors = [door for door in state.values()
                    if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door['is_open']]
                if len(closedDoors)==0:
                    return None, {}
                # Randomly pick a closed door
                self._door = random.choice(closedDoors)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0],doorLoc[1]+1
                # Send message of current action
                self._sendMessage('Moving to ' + self._door['room_name'], agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase=Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR==self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action, {}   
                self._phase=Phase.OPEN_DOOR

            if Phase.OPEN_DOOR==self._phase:
                self._phase=Phase.PLAN_PATH_TO_CLOSED_DOOR
                # Open door
                return OpenDoorAction.__name__, {'object_id':self._door['obj_id']}
    

    def _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _processMessages(self, teamMembers):
        '''
        Process incoming messages and create a dictionary with received messages from each team member.
        '''
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)       
        return receivedMessages

    '''
    Compute the trust belief value based on trust and reputation
    Direct Experiences influence more than Reputation
    '''
    def _computeTrustBeliefs(self, agents):
        trust_beliefs = {}
        for [agent, trust, rep] in agents:
            trust_beliefs[agent] = (2 * float(trust) + float(rep)) / 3
        return trust_beliefs

    '''
    Transform text message into data
    '''
    def _normalizeMessage(self, received):
        """
        Communication protocol:
        Moving to [room_name]
        Opening door of [room_name]
        Searching through [room_name]
        Found goal block [block_visualization] at location [location]
        Picking up goal block [block_visualization] at location [location]
        Dropped goal block [block_visualization] at drop location [location]
        """
        message = received.split()
        if len(message) == 3 and ' '.join(message[:2]) == 'Moving to':
            return MessageType.MOVING, [message[2]]
        elif len(message) == 4 and ' '.join(message[:3]) == 'Opening door of':
            return MessageType.OPENING, [message[3]]
        elif len(message) == 3 and ' '.join(message[:2]) == 'Searching through':
            return MessageType.SEARCHING, [message[2]]
        elif len(message) > 3 and ' '.join(message[:3]) == 'Found goal block':
            return MessageType.FOUND, [received[received.find('{')+1:received.find('}')],
                                       received[received.find('(')+1:received.find(')')]]
        elif len(message) > 3 and ' '.join(message[:3]) == 'Dropped goal block':
            return MessageType.DROPPED, [received[received.find('{')+1:received.find('}')],
                                         received[received.find('(')+1:received.find(')')]]
        elif len(message) > 3 and ' '.join(message[:4]) == 'Picking up goal block':
            return MessageType.PICKING_UP, [received[received.find('{')+1:received.find('}')],
                                            received[received.find('(')+1:received.find(')')]]
        else:
            return MessageType.INVALID, []

    '''
    Verify that the same message is not processed twice
    '''
    def _checkIfMessageAlreadyRecieved(self, received):
        return False
    '''
    Trust mechanism (same) for all Agents
    '''
    def _trustBelief(self, name, members, received):
        # Read (or initialize) memory file
        default = 0.5
        filename = name + '_memory.csv'
        params = ['Agent', 'Trust', 'Reputation']
        agents = []
        try:
            with open(filename, 'r') as mem_file:
                memory = csv.reader(mem_file)
                if os.stat(filename).st_size == 0:
                    raise FileNotFoundError
                next(memory)
                for row in memory:
                    agents.append(row)
                if Counter(members) != Counter([agent[0] for agent in agents]):
                    raise FileNotFoundError
        except FileNotFoundError:
            with open(filename, 'w', newline='') as mem_file:
                memory = csv.writer(mem_file)
                memory.writerow(params)
                agents = []  # clear any previously appended data if agent mismatch
                for member in members:
                    agents.append([member, default, default])
                memory.writerows(agents)

        # Process received messages
        for member in received.keys():
            for message in received[member]:
                message_type, message_data = self._normalizeMessage(message)

        # Save back to memory file
        with open(filename, 'w', newline='') as mem_file:
            memory = csv.writer(mem_file)
            memory.writerow(params)
            memory.writerows(agents)

        return self._computeTrustBeliefs(agents)


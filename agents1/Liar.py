
from hashlib import sha1
from typing import final, List, Dict, Final
import enum, random
from bw4t.BW4TBrain import BW4TBrain
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.messages.message import Message
import pprint
import numpy as np
import math

class Phase(enum.Enum):
    SET_UP_VARIABLES = 0,
    PLAN_PATH_TO_UNSEARCHED_ROOM = 1,
    FOLLOW_PATH_TO_UNSEARCHED_ROOM = 2,
    OPEN_DOOR = 3,
    PLAN_ROOM_EXPLORATION = 4,
    EXPLORE_ROOM = 5,
    PLAN_TO_GOAL_BLOCK = 6,
    FOLLOW_PATH_TO_GOAL_BLOCK=7,
    GRAB_BLOCK = 8,
    PLAN_TO_DROP_ZONE = 9,
    FOLLOW_PATH_TO_DROP_ZONE = 10,
    DROP_BLOCK = 11


class Liar(BW4TBrain):

    def __init__(self, settings:Dict[str,object]):
        super().__init__(settings)
        self._phase = Phase.SET_UP_VARIABLES
        self._teamMembers = []
        

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, 
            action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
        self.roomsToExplore = []
        
        self.receivedInformation = []
        
        # Known data since we have seen it
        self.knownBlocks = []
        self.knownBlockKeys = []
        
        self.knownGoalBlocks = []
        self.blocksDelivered = []
        
        # The blocks we need to collect
        self.collectBlocks = []
        self.collectBlocksBackup = []
        self.dropOffLocations = []
        
        self.blockToGrab = None
        self.locationToDropOff = None
        

        # openedRooms = []

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
        self._trustBlief(self._teamMembers, receivedMessages)
        
        
        
        # print("Extracting the map")
        # map=state.get_traverse_map()
        # width, length = state.get_world_info()['grid_shape']
        # for y in range(length):
        #     for x in range(width):
        #         if map[(x,y)]:
        #             print("*",end="")
        #         else: 
        #             print(" ", end="")
        #     print()
        
        
        while True:
            if Phase.SET_UP_VARIABLES==self._phase:
                
                for key in state.keys():
                    if "Collect_Block" in key:
                        self.collectBlocks.append(state[key])
                        self.collectBlocksBackup.append(state[key])
                    # print(key, state[key])
                self.roomsToExplore = [door for door in state.values()
                    if 'class_inheritance' in door and 'Door' in door['class_inheritance']]      
                self._phase=Phase.PLAN_PATH_TO_UNSEARCHED_ROOM
                
            if Phase.PLAN_PATH_TO_UNSEARCHED_ROOM==self._phase:
                self._navigator.reset_full()
                if len(self.roomsToExplore)==0:
                    print("all rooms are opened")
                    self._phase=Phase.PLAN_TO_GOAL_BLOCK
                    return None, {}
                # Randomly pick a closed door
                self._door = random.choice(self.roomsToExplore)
                self.roomsToExplore.remove(self._door)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0],doorLoc[1]+1
                # Send message of current action
                msgDoor = random.choice([door for door in state.values()
                    if 'class_inheritance' in door and 'Door' in door['class_inheritance'] 
                    and door['room_name'] is not self._door['room_name']])['room_name'] if self.toLieOrNotToLieZetsTheKwestion() else self._door['room_name']
                self._sendMessage('Moving to door of ' + str(msgDoor), agent_name)

                self._navigator.add_waypoints([doorLoc])
                self._phase=Phase.FOLLOW_PATH_TO_UNSEARCHED_ROOM

            if Phase.FOLLOW_PATH_TO_UNSEARCHED_ROOM==self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action, {}   
                self._phase=Phase.OPEN_DOOR

            if Phase.OPEN_DOOR==self._phase:
                self._phase=Phase.PLAN_ROOM_EXPLORATION
                # Open door
                door = random.choice([door for door in state.values()
                    if 'class_inheritance' in door and 'Door' in door['class_inheritance'] 
                    and door['room_name'] is not self._door['room_name']]) if self.toLieOrNotToLieZetsTheKwestion() else self._door
                if not self._door['is_open']:
                    self._sendMessage('Opening door of ' + str(door['room_name']), agent_name)
                return "OpenDoorAction" , {'object_id':self._door['obj_id']}
            
            elif Phase.PLAN_ROOM_EXPLORATION==self._phase:
                self._navigator.reset_full()
                door = self._door
                doorLoc = door['location']
                room = self._getRoomSize(door['room_name'], state)
                waypoints = [(room[1][0]-1,room[1][1])]
                currentX = room[1][0]-1
                currentY = room[1][1]
                self._phase=Phase.EXPLORE_ROOM
                while currentX > room[0][0]+1:
                    if currentY > room[0][1]:
                        waypoints.append((currentX,room[0][1]))
                        currentY = room[0][1]
                    if currentX > room[0][0]+1:
                        waypoints.append((currentX-2,currentY))
                        currentX -=2
                    else:
                        waypoints.append((door['location'][0]))
                    if currentY < room[1][1] :
                        waypoints.append((currentX,room[1][1]))
                        currentY = room[1][1] 
                    if currentX > room[0][0]+1: 
                        waypoints.append((currentX-2,currentY))
                        currentX -=2
                    else:
                        waypoints.append((door['location']))
                self._navigator.add_waypoints(waypoints)
                self._phase=Phase.EXPLORE_ROOM
                
            if Phase.EXPLORE_ROOM==self._phase:

                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                for block in state.keys():
                    if "Block_in" in block:
                        obj_id = state[block]["obj_id"]
                        if obj_id not in self.knownBlockKeys:
                            self.knownBlockKeys.append(obj_id)
                            self.knownBlocks.append(state[block])
                            for goalBlock in self.collectBlocks:
                                if self.sameVizuals(goalBlock, state[block]):
                                    self.knownGoalBlocks.append(state[block])
                                    messageBlock = random.choice([block for block in self.collectBlocks
                                        if goalBlock is not block ]) if self.toLieOrNotToLieZetsTheKwestion() else state[block]
                                    msg = "Found goal block " + str({"size": messageBlock["visualization"]['size'], "shape":  messageBlock["visualization"]['shape'], "colour":  messageBlock["visualization"]['colour']}) + " at location " + str(state[block]['location'])
                                    self._sendMessage(msg, agent_name)
                if action!=None:
                    return action, {}   
                self._phase=Phase.PLAN_PATH_TO_UNSEARCHED_ROOM
            
            if Phase.PLAN_TO_GOAL_BLOCK==self._phase:
                self._navigator.reset_full()
                print("PLANNING 2 GO 2 GOAL BLOCK")
                
                if len(self.collectBlocks) > 0:
                    collectBlock = self.collectBlocks.pop(0)
                else: 
                    return None, {}
                for block in self.knownGoalBlocks:
                    if self.sameVizuals(collectBlock, block):
                        self.blockToGrab = block
                        break
                        
                roomOfBlock = self.blockToGrab['name'].split(' ')[-1]
                # Send message of current action
                msgDoor = random.choice([door for door in state.values()
                    if 'class_inheritance' in door and 'Door' in door['class_inheritance'] 
                    and door['room_name'] is not roomOfBlock])['room_name'] if self.toLieOrNotToLieZetsTheKwestion() else roomOfBlock
                self._sendMessage('Moving to door of ' + str(msgDoor), agent_name) #edit to be able to lie

                
                self._navigator.add_waypoints([self.blockToGrab['location']])
                self._phase=Phase.FOLLOW_PATH_TO_GOAL_BLOCK
            
            if Phase.FOLLOW_PATH_TO_GOAL_BLOCK==self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                # pprint.pprint(action)
                if action!=None:
                    return action, {}   
                self._phase=Phase.GRAB_BLOCK
            
            if Phase.GRAB_BLOCK==self._phase:
                lie = self.toLieOrNotToLieZetsTheKwestion()
                block = self.blockToGrab
                location = self.blockToGrab['location']
                if lie and len(self.knownGoalBlocks) > 1:
                    block = random.choice([block for block in self.knownGoalBlocks
                    if not self.sameVizuals(block, self.blockToGrab)])
                elif lie:
                    location = (math.ceil(random.random() * location[0]), math.ceil(random.random() * location[1])) ## SHOULD BE IMPROVED
                    
                self._sendMessage('Picking up goal block {"size": ' + str(block['visualization']['size'])  
                                  + ', "shape": ' + str(block['visualization']['shape'])
                                  + ', "colour": ' + str(block['visualization']['colour'])
                                  + '} at location ' + str(self.blockToGrab['location']), agent_name)
                self._phase=Phase.PLAN_TO_DROP_ZONE
                return "GrabObject", {'object_id':self.blockToGrab['obj_id'] } 
            
            if Phase.PLAN_TO_DROP_ZONE==self._phase:
                self._navigator.reset_full()
                carriedBlock = self.agent_properties['is_carrying'][0]
                location = (0,0)
                for collecBlock in self.collectBlocksBackup:                 
                    if (self.sameVizuals(collecBlock, carriedBlock)): 
                        location = collecBlock['location']
                self._navigator.add_waypoints([location])
                self.locationToDropOff = location
                self._phase=Phase.FOLLOW_PATH_TO_GOAL_BLOCK
            
            if Phase.FOLLOW_PATH_TO_GOAL_BLOCK==self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action, {}   
                self._phase=Phase.DROP_BLOCK
                
            if Phase.DROP_BLOCK==self._phase:
                carriedBlock = self.agent_properties['is_carrying'][0]
                lie = self.toLieOrNotToLieZetsTheKwestion()
                location = self.locationToDropOff
                block = carriedBlock
                if lie: 
                    if len(self.collectBlocks) > 0:
                        block = random.choice([block for block in self.collectBlocks
                            if (block['visualization']['shape'] is not carriedBlock['visualization']['shape']) or
                            (block['visualization']['colour']   is not carriedBlock['visualization']['colour']) or
                            (block['visualization']['size']     is not carriedBlock['visualization']['size'])])
                        location = block['location']
                    else: 
                        block = self.knownBlocks[0]
                self._sendMessage('Dropped goal block {"size": ' + str(block['visualization']['size'])  
                                  + ', "shape": ' + str(block['visualization']['shape'])
                                  + ', "colour": ' + str(block['visualization']['colour'])
                                  + '} at drop location ' + str(location), agent_name)
                self._phase=Phase.PLAN_TO_GOAL_BLOCK
                return "DropObject", {'object_id':self.agent_properties['is_carrying'][0]['obj_id'] }            
                
    def sameVizuals(self, block1, block2):
        return (block1['visualization']['shape'] == block2['visualization']['shape'] and
                    block1['visualization']['colour'] == block2['visualization']['colour'] and
                    block1['visualization']['size'] == block2['visualization']['size'])  
                
    def toLieOrNotToLieZetsTheKwestion(self):
        lie= random.random() < 0.8
        # if lie:
        #     self._sendMessage("My next message is a lie", self.state[self.agent_id]['obj_id'])
        return lie
    
    def _getRoomSize(self, room, state:State):
        startX = startY = endX = endY = None
        for roomTile in state.get_room_objects(room):
            if 'area' in roomTile['name']:
                if endX is None or roomTile['location'][0] > endX:
                    endX = roomTile['location'][0]
                if endY is None or roomTile['location'][0] > endY:
                    endY = roomTile['location'][1]
                if startX is None or roomTile['location'][0] < startX:
                    startX = roomTile['location'][0]
                if startY is None or roomTile['location'][0] < startY:
                    startY = roomTile['location'][1]
        return [(startX, startY), (endX, endY)]
        
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

    def _trustBlief(self, member, received):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''
        # You can change the default value to your preference
        default = 0.5
        trustBeliefs = {}
        for member in received.keys():
            trustBeliefs[member] = default
        for member in received.keys():
            for message in received[member]:
                if 'Found' in message and 'colour' not in message:
                    trustBeliefs[member]-=0.1
                    break
        return trustBeliefs
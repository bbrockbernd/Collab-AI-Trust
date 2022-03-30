
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
    DROP_BLOCK_AT_COLLECTION_POINT = 11
    DROP_BLOCK_WEST_OF_COLLECTION_POINT = 12


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
        self.knownBlocks = {}
        
        # The blocks we need to collect
        self.collectBlocks = {}
        
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
        
        
        while True:
            if Phase.SET_UP_VARIABLES==self._phase:
                for key in state.keys():
                    if "Collect_Block" in key:
                        self.collectBlocks[key] = state[key]
                        self.collectBlocks[key]['drop_actions'] = []
                        self.collectBlocks[key]['is_delivered_confirmed'] = False
                        self.collectBlocks[key]['is_delivered_by_me'] = False # dropActions = {'agent': None, 'number': None}
                self.roomsToExplore = [door for door in state.values()
                    if 'class_inheritance' in door and 'Door' in door['class_inheritance']] 
                         
                self._phase=Phase.PLAN_PATH_TO_UNSEARCHED_ROOM
                
            if Phase.PLAN_PATH_TO_UNSEARCHED_ROOM==self._phase:
                if len(self.roomsToExplore)>0:
                    self._planPathToUnsearchedRoom() 
                    self._sendMovingToDoorMessage(state, self._door)
                    self._phase=Phase.FOLLOW_PATH_TO_UNSEARCHED_ROOM
                else:
                    self._phase=Phase.PLAN_TO_GOAL_BLOCK
                

            if Phase.FOLLOW_PATH_TO_UNSEARCHED_ROOM==self._phase:
                self.updateBlocks(state)
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action, {}   
                self._phase=Phase.OPEN_DOOR

            if Phase.OPEN_DOOR==self._phase:
                self._phase=Phase.PLAN_ROOM_EXPLORATION                
                if not self._door['is_open']:
                    for block in state.keys():
                        if "door" in block and state[block]["obj_id"] == self._door['obj_id'] and not state[block]["is_open"]:
                            self._sendDoorOpenMessage(state)
                            return "OpenDoorAction" , {'object_id':self._door['obj_id']}  
            
            elif Phase.PLAN_ROOM_EXPLORATION==self._phase:
                self._roomExplorationWayPoints(state)
                self._phase=Phase.EXPLORE_ROOM
                self.sendExploringMessage(state)
                
            if Phase.EXPLORE_ROOM==self._phase:
                self.updateBlocks(state)
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action, {}   
                self._phase=Phase.PLAN_PATH_TO_UNSEARCHED_ROOM
            
            if Phase.PLAN_TO_GOAL_BLOCK==self._phase:
                possible = self._planPathToGoalBlock()
                if possible == False:
                    return None, {}
                
                roomOfBlock = self.blockToGrab['name'].split(' ')[-1]
                self._sendMovingToDoorMessage(state, roomOfBlock)

                self._phase=Phase.FOLLOW_PATH_TO_GOAL_BLOCK
            
            if Phase.FOLLOW_PATH_TO_GOAL_BLOCK==self._phase:
                self.updateBlocks(state)
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action, {}   
                self._phase=Phase.GRAB_BLOCK
            
            if Phase.GRAB_BLOCK==self._phase:
                
                blocks = self.detectBlocksAround(state)
                ids = []
                for block in blocks:
                    ids.append(block['obj_id'])
                if self.blockToGrab['obj_id'] not in ids:
                    #BLOCK NOT ON LAST KNOWN LOCATION
                    del self.knownBlocks[self.blockToGrab['obj_id']]
                    self._phase = Phase.PLAN_TO_GOAL_BLOCK
                    continue
                        
                
                self._sendGrabBlockMessage(state)
                self._phase=Phase.PLAN_TO_DROP_ZONE
                print("GRAPPING: ", self.blockToGrab['obj_id'])
                return "GrabObject", {'object_id':self.blockToGrab['obj_id'] } 
            
            if Phase.PLAN_TO_DROP_ZONE==self._phase:
                if(len(self.agent_properties['is_carrying']) == 0):
                    #NO BLOCKS BRAPPED
                    self._phase = Phase.PLAN_TO_GOAL_BLOCK
                    continue
                
                
                self._planPathToDropOff()
                self._phase=Phase.FOLLOW_PATH_TO_DROP_ZONE
            
            if Phase.FOLLOW_PATH_TO_DROP_ZONE==self._phase:
                self.updateBlocks(state)
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action, {}   
                self._phase=Phase.DROP_BLOCK_AT_COLLECTION_POINT
                
            if Phase.DROP_BLOCK_AT_COLLECTION_POINT==self._phase:
                if not self.checkGoalBlockPresent(state):
                    self.dropGoalBlockAtCollectPoint(state)
                    self.msgAboutDropLocation(state) 
                    return "DropObject", {'object_id':self.agent_properties['is_carrying'][0]['obj_id'] }            
                else: 
                    for collectBlock in self.collectBlocks.values():
                        if collectBlock['location'] == state[self.agent_id]['location']:    
                            self.collectBlocks[collectBlock['obj_id']]['is_delivered_confirmed'] = True
                    self._phase=Phase.DROP_BLOCK_WEST_OF_COLLECTION_POINT
                    self.msgAboutDropLocation(state) 
                    return "MoveWest", {}
                
            if Phase.DROP_BLOCK_WEST_OF_COLLECTION_POINT==self._phase:
                self._phase=Phase.PLAN_TO_GOAL_BLOCK
                carriedBlockId = self.agent_properties['is_carrying'][0]['obj_id']
                self.knownBlocks[carriedBlockId]['is_delivered_by_me'] = True
                self.knownBlocks[carriedBlockId]['is_delivered'] = True
                return "DropObject", {'object_id':carriedBlockId}  
    
    def _planPathToUnsearchedRoom(self):
        self._navigator.reset_full()
        # Randomly pick a closed door
        self._door = self.roomsToExplore[0]#random.choice(self.roomsToExplore)
        self.roomsToExplore.remove(self._door)
        doorLoc = self._door['location']
        # Location in front of door is south from door
        doorLoc = doorLoc[0],doorLoc[1]+1
        # Send message of current action
        self._navigator.add_waypoints([doorLoc])  
        
    def getBlockToGrab(self):
        for _collectBlock in self.collectBlocks.values():
            if not _collectBlock['is_delivered_by_me'] and not _collectBlock['is_delivered_confirmed']:
                collectBlock = _collectBlock
                
                if collectBlock is None:
                    return None
        
                for block_id in self.knownBlocks:
                    block = self.knownBlocks[block_id]
                    if self.knownBlocks[block_id]['isGoalBlock'] and self.knownBlocks[block_id]['is_delivered'] == False and self.sameVizuals(collectBlock, block):
                        self.blockToGrab = block
                        return block
                    
    def _planPathToGoalBlock(self):
        self._navigator.reset_full()
        collectBlock = self.getBlockToGrab()
        if collectBlock is None:
            return None, {}
        self._navigator.add_waypoints([self.blockToGrab['location']])
        
              
    def _planPathToDropOff(self):
        self._navigator.reset_full()
        carriedBlock = self.agent_properties['is_carrying'][0]
        location = (0,0)
        for collectBlock in self.collectBlocks.values():                 
            if (self.sameVizuals(collectBlock, carriedBlock)): 
                location = collectBlock['location']
        self._navigator.add_waypoints([location])
        self.locationToDropOff = location 
     
    def detectBlocksAround(self, state:State):
        result = []
        for block in state.keys():
            if "Block_in" in block:
                result.append(state[block])
        return result
    
    def addNewBlock(self, state:State, block):
        obj_id = block['obj_id']
        if obj_id not in self.knownBlocks.keys():
            self.knownBlocks[obj_id] = block
            self.knownBlocks[obj_id]["isGoalBlock"] = False
            for collectBlock in self.collectBlocks.values():
                if self.sameVizuals(collectBlock, block):
                    self.knownBlocks[obj_id]["isGoalBlock"] = True
                    self.knownBlocks[obj_id]['is_delivered'] = False
                    self.knownBlocks[obj_id]['is_delivered_confirmed'] = False
                    self.knownBlocks[obj_id]['is_delivered_by_me'] = False                    
                    self.sendGoalBlockFoundMessage(state, collectBlock)
                
                
    '''
    Detects if there are any new block in the reachable area
    '''
    def detectNewBlocks(self, state:State):
        for block in self.detectBlocksAround(state):
            self.addNewBlock(state, block)
    
    def sendExploringMessage(self, state:State):
        msg = random.choice([door for door in state.values()
            if 'class_inheritance' in door and 'Door' in door['class_inheritance'] 
            and door['room_name'] is not self._door['room_name']])['room_name'] if self.toLieOrNotToLieZetsTheKwestion() else self._door['room_name']
        self._sendMessage("Searching through "  + str(msg), state[self.agent_id]['obj_id']) 
        
    def _sendMovingToDoorMessage(self, state:State, correctDoor):       
        msg = random.choice([door for door in state.values()
            if 'class_inheritance' in door and 'Door' in door['class_inheritance'] 
            and door['room_name'] is not correctDoor])['room_name'] if self.toLieOrNotToLieZetsTheKwestion() else correctDoor
        self._sendMessage('Moving to ' + str(msg), state[self.agent_id]['obj_id'])
            
    def _sendDoorOpenMessage(self, state:State):
        door = random.choice([door for door in state.values()
                    if 'class_inheritance' in door and 'Door' in door['class_inheritance'] 
                    and door['room_name'] is not self._door['room_name']]) if self.toLieOrNotToLieZetsTheKwestion() else self._door
        self._sendMessage('Opening door of ' + str(door['room_name']), state[self.agent_id]['obj_id'])
         
    def sendGoalBlockFoundMessage(self, state:State, block):
        lieOptions = [otherBlock for otherBlock in self.collectBlocks.values()
                    if not self.sameVizuals(block, otherBlock)]
        if len(lieOptions) > 0:
            lie = random.choice(lieOptions)
            location = str(state[block['obj_id']]['location'])
        else: 
            lie = state[block['obj_id']]
            location = random.choice([otherBlock for otherBlock in self.knownBlocks.values()])['location']
        messageBlock = lie if self.toLieOrNotToLieZetsTheKwestion() else state[block['obj_id']] 
        msg = "Found goal block " + str({"size": messageBlock["visualization"]['size'], "shape":  messageBlock["visualization"]['shape'], "colour":  messageBlock["visualization"]['colour']}) + " at location " + location
        self._sendMessage(msg, state[self.agent_id]['obj_id'])
    
    def _sendGrabBlockMessage(self, state:State):
        lie = self.toLieOrNotToLieZetsTheKwestion()
        block = self.blockToGrab
        location = self.blockToGrab['location']
        if lie and len(self.knownBlocks) > 1:
            block = random.choice([block for block_id in self.knownBlocks
            if not self.sameVizuals(self.knownBlocks[block_id], self.blockToGrab)])
        elif lie:
            location = (math.ceil(random.random() * location[0]), math.ceil(random.random() * location[1])) ## SHOULD BE IMPROVED
            
        self._sendMessage('Picking up goal block {"size": ' + str(block['visualization']['size'])  
                            + ', "shape": ' + str(block['visualization']['shape'])
                            + ', "colour": ' + str(block['visualization']['colour'])
                            + '} at location ' + str(self.blockToGrab['location']), state[self.agent_id]['obj_id'])           
                
    def updateBlock(self, block):
        obj_id = block['obj_id']
        if obj_id in self.knownBlocks.keys():
            self.knownBlocks[obj_id]['location'] = block['location']
            for collectBlock in self.collectBlocks.values():
                if self.sameVizuals(block, collectBlock):
                    if block['location'] == collectBlock['location']:
                        self.knownBlocks[obj_id]['is_delivered'] = True
                        self.knownBlocks[obj_id]['is_delivered_confirmed'] = True
                                                    
            
    '''
    Update existing blocks
    '''
    def updateBlocks(self, state:State):
        for block in self.detectBlocksAround(state):
            self.addNewBlock(state, block)
            self.updateBlock(block)
                               
    
    
    def msgAboutDropLocation(self, state:State):
        carriedBlock = self.agent_properties['is_carrying'][0]
        lie = self.toLieOrNotToLieZetsTheKwestion()
        location = state[self.agent_id]['location']
        block = carriedBlock
        
        if lie: 
            if len(self.collectBlocks) > 0:
                block = random.choice([block for block in self.collectBlocks.values()
                    if  (block['visualization']['shape']  is not carriedBlock['visualization']['shape']) or
                        (block['visualization']['colour'] is not carriedBlock['visualization']['colour']) or
                        (block['visualization']['size']   is not carriedBlock['visualization']['size'])])
                
        self._sendMessage('Dropped goal block {"size": ' + str(block['visualization']['size'])  
                            + ', "shape": ' + str(block['visualization']['shape'])
                            + ', "colour": ' + str(block['visualization']['colour'])
                            + '} at drop location ' + str(location), state[self.agent_id]['obj_id'])
            
    def checkGoalBlockPresent(self, state:State):
        for block in state.keys():
            if "Block_in" in block and state[block]["location"] == state[self.agent_id]['location']:
                return True
        return False
    
    def dropGoalBlockAtCollectPoint(self, state:State):
        carriedBlock = self.agent_properties['is_carrying'][0]

        self._phase=Phase.PLAN_TO_GOAL_BLOCK
    
        self.knownBlocks[carriedBlock['obj_id']]['is_delivered_by_me'] = True
        self.knownBlocks[carriedBlock['obj_id']]['is_delivered'] = True
        for key in self.collectBlocks.keys():
            if self.collectBlocks[key]['location'] == state[self.agent_id]['location']:
                self.collectBlocks[key]['is_delivered_by_me'] = True
                self.collectBlocks[key]['is_delivered_confirmed'] = True
                break   
      
    def sameVizuals(self, block1, block2):
        return (block1['visualization']['shape'] == block2['visualization']['shape'] and
                    block1['visualization']['colour'] == block2['visualization']['colour'] and
                    block1['visualization']['size'] == block2['visualization']['size'])  
                
    def toLieOrNotToLieZetsTheKwestion(self):
        lie= random.random() < 0.8
        # if lie:
        #     self._sendMessage("My next message is a lie", self.state[self.agent_id]['obj_id'])
        return lie
    
    def _roomExplorationWayPoints(self, state:State):
        self._navigator.reset_full()
        door = self._door
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
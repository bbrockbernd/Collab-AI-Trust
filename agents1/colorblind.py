import random
from typing import Dict
import enum

from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.messages.message import Message

from agents1.BW4TBaselineAgent import BaseLineAgent


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
    PLAN_PATH_TO_VERIFY_COLLECTION = 11,
    FOLLOW_PATH_TO_VERIFY_COLLECTION = 12


class Colorblind(BaseLineAgent):

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
        receivedMessages = super()._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        super()._trustBelief(agent_name, self._teamMembers, receivedMessages, state)
        
        
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
                if len(self.roomsToExplore)>0 and not self._possibleToPlanPathToGoalBlock():
                    self._planPathToUnsearchedRoom() 
                    self._sendMovingToDoorMessage(state, self._door['room_name'])
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
                possible = self._possibleToPlanPathToGoalBlock()
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
                self._phase=Phase.PLAN_PATH_TO_VERIFY_COLLECTION
                self.processDropGoalBlockAtCollectPoint(state)
                self.msgAboutDropLocation(state)
                return "DropObject", {'object_id':self.agent_properties['is_carrying'][0]['obj_id'] }    
            
            if Phase.PLAN_PATH_TO_VERIFY_COLLECTION==self._phase:
                self.updateBlocks(state)
                self._navigator.reset_full()
                locations = []
                for collectBlock in self.collectBlocks.values():                 
                    locations.append(collectBlock['location'])
                self._navigator.add_waypoints(locations)
                self._phase=Phase.FOLLOW_PATH_TO_VERIFY_COLLECTION
                
            if Phase.FOLLOW_PATH_TO_VERIFY_COLLECTION==self._phase:
                self.updateBlocks(state)
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action, {}   
                
                elif len (self.roomsToExplore) > 0:
                    self._phase=Phase.PLAN_PATH_TO_UNSEARCHED_ROOM
                else:
                    self._phase=Phase.PLAN_TO_GOAL_BLOCK
    
    def _planPathToUnsearchedRoom(self):
        self._navigator.reset_full()
        # Randomly pick a closed door
        self._door = random.choice(self.roomsToExplore)
        self.roomsToExplore.remove(self._door)
        doorLoc = self._door['location']
        # Location in front of door is south from door
        doorLoc = doorLoc[0],doorLoc[1]+1
        # Send message of current action
        self._navigator.add_waypoints([doorLoc])  
        
    def getBlockToGrab(self):
        for _collectBlock in self.collectBlocks.values():
            if not _collectBlock['is_delivered_by_me'] or not _collectBlock['is_delivered_confirmed']:
                collectBlock = _collectBlock
                
                if collectBlock is None:
                    return None
                ids = list(self.knownBlocks.keys())
                ids.sort(reverse=True)
                for id in ids:
                    block = self.knownBlocks[id]
                    if block['isGoalBlock'] and block['is_delivered'] == False and self.sameVizuals(collectBlock, block):
                        self.blockToGrab = block
                        return block
                return None
                    
    def _possibleToPlanPathToGoalBlock(self):
        self._navigator.reset_full()
        self.blockToGrab = self.getBlockToGrab() #MAYBY MOVE SETTING
        if self.blockToGrab is None:
            return False
        self._navigator.add_waypoints([self.blockToGrab['location']])
        return True
    
    def _getTargetLocation(self, block):
        location = (-1, -1)
        ids = list(self.collectBlocks.keys())
        ids.sort()
        
        for name in ids:            
            if self.sameVizuals(self.collectBlocks[name], block):
                location = self.collectBlocks[name]["location"]
                break
        
        location = (location[0]-1, location[1])
        return location
        
              
    def _planPathToDropOff(self):
        self._navigator.reset_full()
        carriedBlock = self.agent_properties['is_carrying'][0]
        location = self._getTargetLocation(carriedBlock)
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
            if self.knownBlocks[obj_id]["isGoalBlock"]:
                self.sendGoalBlockFoundMessage(state, block)
                
                
    '''
    Detects if there are any new block in the reachable area
    '''
    def detectNewBlocks(self, state:State):
        for block in self.detectBlocksAround(state):
            self.addNewBlock(state, block)
    
    def sendExploringMessage(self, state:State):
        msg = self._door['room_name']
        if type(msg) == str:
            super()._sendMessage("Searching through "  + msg, state[self.agent_id]['obj_id'])
        else:
            super()._sendMessage("Searching through "  + msg["door_name"], state[self.agent_id]['obj_id']) 
        
    def _sendMovingToDoorMessage(self, state:State, correctDoor):       
        msg = correctDoor
        super()._sendMessage('Moving to ' + str(msg), state[self.agent_id]['obj_id'])
            
    def _sendDoorOpenMessage(self, state:State):
        door = self._door
        super()._sendMessage('Opening door of ' + str(door['room_name']), state[self.agent_id]['obj_id'])
         
    def sendGoalBlockFoundMessage(self, state:State, block):
        messageBlock = state[block['obj_id']] 
        location = str(state[block['obj_id']]['location'])
        msg = "Found goal block " + str({"size": messageBlock["visualization"]['size'], "shape":  messageBlock["visualization"]['shape'], "colour":  "?"}) + " at location " + location
        super()._sendMessage(msg, state[self.agent_id]['obj_id'])
    
    def _sendGrabBlockMessage(self, state:State):
        block = self.blockToGrab
        location = self.blockToGrab['location']
            
        super()._sendMessage("Picking up goal block " + str ({"size": block['visualization']['size'], 
                                                                          "shape": block['visualization']['shape'],
                                                                          "colour": "?"})+ " at location " + str(self.blockToGrab['location']), state[self.agent_id]['obj_id'])           
                
    def updateBlock(self, block):
        obj_id = block['obj_id']
        if obj_id in self.knownBlocks.keys():
            self.knownBlocks[obj_id]['location'] = block['location']
            
            #It is placed where we want it
            if self._getTargetLocation(self.knownBlocks[obj_id]) == self.knownBlocks[obj_id]['location']:
                self.knownBlocks[obj_id]['is_delivered'] = True
                self.knownBlocks[obj_id]['is_delivered_confirmed'] = True
                return
                
            #it is placed on a dropzone we asume it is valid
            for collectBlock in self.collectBlocks.values():
                if self.sameVizuals(block, collectBlock):
                    if block['location'] == collectBlock['location']:
                        self.knownBlocks[obj_id]['is_delivered'] = True
                        self.knownBlocks[obj_id]['is_delivered_confirmed'] = True
                        return

                                                    
            
    '''
    Update existing blocks
    '''
    def updateBlocks(self, state:State):
        for block in self.detectBlocksAround(state):
            self.addNewBlock(state, block)
            self.updateBlock(block)
                               
    
    
    def msgAboutDropLocation(self, state:State):
        carriedBlock = self.agent_properties['is_carrying'][0]
        location = state[self.agent_id]['location']
        block = carriedBlock
        
        super()._sendMessage("Dropped goal block " + str({"size": block['visualization']['size'], 
                                                         "shape": block['visualization']['shape'],
                                                         "colour": "?" }) + " at drop location " + str(location), state[self.agent_id]['obj_id'])
            
    def checkGoalBlockPresent(self, state:State):
        for block in state.keys():
            if "Block_in" in block and state[block]["location"] == state[self.agent_id]['location']:
                return True
        return False
    
    def processDropGoalBlockAtCollectPoint(self, state:State):
        carriedBlock = self.agent_properties['is_carrying'][0]
        self.knownBlocks[carriedBlock['obj_id']]['is_delivered_by_me'] = True
        self.knownBlocks[carriedBlock['obj_id']]['is_delivered'] = True
        for key in self.collectBlocks.keys():
            if self.collectBlocks[key]['location'] == state[self.agent_id]['location']:
                # self.collectBlocks[key]['is_delivered_by_me'] = True
                # self.collectBlocks[key]['is_delivered_confirmed'] = True
                break   
      
    def sameVizuals(self, block1, block2):
        return (block1['visualization']['shape'] == block2['visualization']['shape'] and
                    block1['visualization']['size'] == block2['visualization']['size'])
    
    def _validateBlock(self, location, color: str, shape: int): 
        possible_blocks = []
        for block in self.knownBlocks.values():
            if (block['location'] == location):
                possible_blocks.append(block)
                if block['visualization']['shape'] == shape: 
                    return 1
        if len(possible_blocks) == 0:
            return 0
        return -1
    
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
        
    
    
    def validateBlock(self, location, color: str, shape: int): 
        possible_blocks = []
        for block in self.knownBlocks.values():
            if (block['location'] == location):
                possible_blocks.append(block)
                if (block['visualization']['shape'] == shape):
                        return 1
        if len(possible_blocks) == 0:
            return 0
        return -1
    
    def _getRoomSize(self, room, state:State):
        startX = startY = endX = endY = None
        for roomTile in state.get_room_objects(room):
            if 'area' in roomTile['name']:
                if endX is None or roomTile['location'][0] > endX:
                    endX = roomTile['location'][0]
                if endY is None or roomTile['location'][1] > endY:
                    endY = roomTile['location'][1]
                if startX is None or roomTile['location'][0] < startX:
                    startX = roomTile['location'][0]
                if startY is None or roomTile['location'][1] < startY:
                    startY = roomTile['location'][1]
        return [(startX, startY), (endX, endY)]
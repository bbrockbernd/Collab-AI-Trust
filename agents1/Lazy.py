import ast
from typing import List, Dict
import enum, random
from agents1.BW4TBaselineAgent import BaseLineAgent
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject


class Phase(enum.Enum):
    INITIALIZE = 0
    WHAT_TO_DO = 1
    CALCULATING = 2
    MOVING = 3
    OPEN_DOOR = 4
    PICKUP = 5
    DROP = 6
    PLAN_ROOM_EXPLORE = 7
    EXPLORE_ROOM = 8


class Mode(enum.Enum):
    EXPLORING = 0
    GOAL = 1

class MyDropPoint:
    def __init__(self, obj_id: str, shape: int, color: str, location: (int, int)):
        self.shape = shape
        self.location = location
        self.color = color
        self.obj_id = obj_id
        self.completed = False
        self.goals: List[MyBlock] = []

        # Not sure if this is unique
        self.myid = str(shape) + color


class MyBlock:
    def __init__(self, block_obj: {}, room: str, isGoal=False):
        self.shape = block_obj['visualization']['shape']
        self.color = block_obj['visualization']['colour']
        self.isGoal = isGoal
        self.location = block_obj['location']
        self.dropPoint = None
        self.completed = False
        self.room = room
        self.myid = str(self.shape) + self.color
        self.obj_id = block_obj['obj_id']
        self.visualization = block_obj['visualization']


    def set_drop_point(self, dp: MyDropPoint):
        self.dropPoint = dp


class MyRoom:

    def __init__(self, roomName: str, doorDict: {}, roomContents: []):
        self.name = roomName
        self.blocks = []
        self.explored = False
        self.doorLoc = doorDict['location']
        self.doorOpen = False
        self.doorId = doorDict['obj_id']
        self.roomSquares = [room['location'] for room in roomContents if 'area' in room['name']]

    def getGoals(self):
        return [block for block in self.blocks if block.isGoal]


class MyWorld:
    def __init__(self):
        self.rooms = {}
        self.dropPoints: List[MyDropPoint] = []
        self.doorLocs = {}
        self.blocks = {}

    def addRoom(self, roomName: str, doorDict: {}, roomContents: []):
        room = MyRoom(roomName, doorDict, roomContents)
        self.rooms[room.name] = room
        self.doorLocs[room.doorLoc] = room.name

    def getRoom(self, name: str) -> MyRoom:
        return self.rooms[name]

    def getUnexploredRooms(self) -> List[MyRoom]:
        return [room for room in self.rooms.values() if not room.explored]

    def addBlock(self, block: MyBlock):
        if block.obj_id in self.blocks.keys():
            return
        self.blocks[block.obj_id] = block
        self.rooms[block.room].blocks.append(block)

        for dp in self.dropPoints:
            if dp.myid == block.myid:
                block.dropPoint = dp
                block.isGoal = True
                dp.goals.append(block)

    def removeBlock(self, block: MyBlock):
        del self.blocks[block.obj_id]
        self.getRoom(block.room).blocks.remove(block)

        for dp in self.dropPoints:
            if dp.myid == block.myid:
                dp.goals.remove(block)


    def getGoals(self) -> [MyBlock]:
        for dp in self.dropPoints:
            if dp.completed:
                continue
            return [block for block in dp.goals if not block.completed]

        return []


    def addDropPoint(self, dp: MyDropPoint):
        self.dropPoints.append(dp)


    def am_i_at_door(self, location: (int, int)) -> str:
        try:
            return self.doorLocs[location]
        except KeyError:
            return ""


class Lazy(BaseLineAgent):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.INITIALIZE
        self._mode = Mode.EXPLORING
        self._destination = (0, 0)
        self._dest_id = ""
        self._teamMembers = []
        self._world = MyWorld()
        self._current_state: State = None
        self._location = (0, 0)
        self._current_door_id = ""
        self._inventory = None
        self._checked_locations = []
        self._droppoint = None

        self._quitting = False
        self._counter = 0

        self.doStuff = {
            Phase.INITIALIZE: self.init_vars,
            Phase.WHAT_TO_DO: self.what_to_do,
            Phase.CALCULATING: self.calculate,
            Phase.MOVING: self.moving,
            Phase.OPEN_DOOR: self.open_door,
            Phase.EXPLORE_ROOM: self.explore_room,
            Phase.PLAN_ROOM_EXPLORE: self.plan_room_explore,
            Phase.DROP: self.drop_block,
            Phase.PICKUP: self.pickup_block,
        }

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set,
                                    algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):
        return state

    '''
    Check if current mode is still the best (and introduce lazyness
    '''

    def what_am_i_doing(self):
        self._counter += 1

        if self._mode == Mode.GOAL and self._inventory is not None:
            if self._inventory.dropPoint.completed:
                self._phase = Phase.DROP

        if self._mode is Mode.GOAL:
            return

        if len(self._world.getGoals()) > 0 and self._phase is not Phase.EXPLORE_ROOM:
            super()._sendMessage('Quitting current task', self.agent_id)
            self._phase = Phase.WHAT_TO_DO

        if self._counter > 7 and self._quitting and self._phase is not Phase.EXPLORE_ROOM:
            super()._sendMessage('Quitting current task', self.agent_id)
            self._phase = Phase.WHAT_TO_DO


    def init_vars(self) -> (str, {}):
        state = self._current_state
        for room in state.get_all_room_names():
            if 'room' not in room:
                continue
            self._world.addRoom(room, state.get_room_doors(room)[0], state.get_room(room))
        for key in state.keys():
            if "Collect_Block" in key:
                location = state[key]['location']
                colour = state[key]['visualization']['colour']
                shape = state[key]['visualization']['shape']
                self._world.addDropPoint(MyDropPoint(key, shape, colour, location))

        self._world.dropPoints.sort(key= lambda dp: dp.obj_id)

        return self.next(Phase.WHAT_TO_DO)

    def what_to_do(self) -> (str, {}):
        self._counter = 0
        self._quitting = False
        if len(self._world.getGoals()) > 0:
            self._mode = Mode.GOAL
            self._destination = self._world.getGoals()[0].location
            self._dest_id = self._world.getGoals()[0].obj_id
            print("Going to Goal")
        else:
            self._mode = Mode.EXPLORING
            room = random.choice(self._world.getUnexploredRooms())
            self._destination = room.doorLoc
            self._destination = (self._destination[0], self._destination[1] + 1)
            self._dest_id = room.name
            super()._sendMessage(f'Moving to {room.name}', self.agent_id)

            self._quitting = random.choice([True, False])

        return self.next(Phase.CALCULATING)

    def calculate(self) -> (str, {}):
        self._navigator.reset_full()
        self._navigator.add_waypoint(self._destination)

        return self.next(Phase.MOVING)

    def moving(self) -> (str, {}):
        self._state_tracker.update(self._current_state)
        action = self._navigator.get_move_action(self._state_tracker)
        room_id = self._world.am_i_at_door((self._location[0],
                                            self._location[1] - 1))

        if room_id != "" and not self._world.getRoom(room_id).doorOpen:
            self._current_door_id = self._world.getRoom(room_id).doorId
            return self.next(Phase.OPEN_DOOR)

        if action != None:
            return action, {}

        if self._mode == Mode.EXPLORING:
            return self.next(Phase.PLAN_ROOM_EXPLORE)

        if self._mode == Mode.GOAL and self._inventory is None:
            return self.next(Phase.PICKUP)

        if self._mode == Mode.GOAL and self._inventory is not None:
            return self.next(Phase.DROP)

    def pickup_block(self) -> (str, {}):

        block = self._world.blocks[self._dest_id]
        if block.obj_id not in self._current_state.keys():
            self._world.removeBlock(block)
            return self.next(Phase.WHAT_TO_DO)


        self._dest_id = block.dropPoint.obj_id
        self._destination = block.dropPoint.location
        self._phase = Phase.CALCULATING
        self._inventory = block

        super()._sendMessage(f'Picking up goal block {block.visualization} at location {block.location}', self.agent_id)
        return GrabObject.__name__, {'object_id': block.obj_id}

    def drop_block(self) -> (str, {}):
        block = self._inventory
        self._inventory = None
        block.completed = True
        block.dropPoint.completed = True
        self._phase = Phase.WHAT_TO_DO

        super()._sendMessage(f'Dropped goal block {block.visualization} at drop location {self._location}', self.agent_id)
        return DropObject.__name__, {'object_id': block.obj_id}

    def open_door(self) -> (str, {}):
        self._phase = Phase.MOVING

        super()._sendMessage(f"Opening door of {self._current_door_id.split('_')[0]}_{self._current_door_id.split('_')[1]}", self.agent_id)
        return OpenDoorAction.__name__, {'object_id': self._current_door_id}

    def plan_room_explore(self) -> (str, {}):

        assert 'room' in self._dest_id
        roomSquares = self._world.getRoom(self._dest_id).roomSquares
        self._navigator.reset_full()
        self._navigator.add_waypoints(roomSquares)

        super()._sendMessage(f"Searching through {self._dest_id}", self.agent_id)
        return self.next(Phase.EXPLORE_ROOM)

    def check_surroundings(self):
        for key in self._current_state.keys():
            if 'Block_in' in key and key not in self._world.blocks:
                block_obj = self._current_state[key]
                block = MyBlock(block_obj, self._dest_id)
                self._world.addBlock(block)
                if block.isGoal:
                    super()._sendMessage(f'Found goal block {block.visualization} at location {block.location}', self.agent_id)

    def explore_room(self) -> (str, {}):
        self.check_surroundings()
        self._checked_locations.append(self._location)

        self._state_tracker.update(self._current_state)
        action = self._navigator.get_move_action(self._state_tracker)

        if action != None:
            return action, {}

        self._world.getRoom(self._dest_id).explored = True
        return self.next(Phase.WHAT_TO_DO)


    def next(self, phase: Phase = None) -> (str, {}):
        if phase is not None:
            self._phase = phase
        return self.doStuff[self._phase]()

    def decide_on_bw4t_action(self, state: State):
        self._current_state = state
        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
                # Process messages from team members
        receivedMessages = super()._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        super()._trustBelief(agent_name, self._teamMembers, receivedMessages, state)

        self._location = self._current_state.get_self()['location']

        if self._phase != Phase.INITIALIZE:
            for room in state.get_all_room_names():
                if 'room' not in room:
                    continue
                door = state.get_room_doors(room)[0]
                self._world.getRoom(room).doorOpen = door['is_open']

        for agent in receivedMessages.keys():
            for msg in receivedMessages[agent]:
                message = msg.split()
                if len(message) > 3 and ' '.join(message[:3]) == 'Dropped goal block':
                    visualization_string = "{" + msg.split('{')[1].split('}')[0] + "}"
                    visualization = ast.literal_eval(visualization_string)

                    location_string = "(" + msg.split('(')[1].split(')')[0] + ")"
                    location = ast.literal_eval(location_string)

                    for dp in self._world.dropPoints:
                        if dp.shape == visualization['shape'] and dp.color == visualization['colour'] and not dp.completed and dp.location == location:
                            dp.completed = True
                            break
                        if not dp.completed:
                            break




        self.what_am_i_doing()

        return self.next(self._phase)

    def validateBlock(self, location: (int, int), color: str, shape: int) -> int:
        if location not in self._checked_locations:
            return 0
        for block in self._world.blocks.values():
            if block.location == location and block.color == color and block.shape == shape:
                return 1
        return -1


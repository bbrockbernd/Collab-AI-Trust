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


class Lazy(BW4TBrain):

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
        self._agent_id = ""

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
        if Mode.GOAL:
            return

        if len(self._world.getGoals()) > 0 and self._phase is not Phase.EXPLORE_ROOM:
            self._sendMessage('Quitting current task', self._agent_id)
            self._phase = Phase.WHAT_TO_DO

        if self._counter == 10 and self._quitting and self._phase is not Phase.EXPLORE_ROOM:
            self._sendMessage('Quitting current task', self._agent_id)
            self._phase = Phase.WHAT_TO_DO


    def init_vars(self) -> (str, {}):
        state = self._current_state
        self._agent_id = state
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
            self._sendMessage(f'Moving to {room.name}', self._agent_id)

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
        self._dest_id = block.dropPoint.obj_id
        self._destination = block.dropPoint.location
        self._phase = Phase.CALCULATING
        self._inventory = block

        self._sendMessage(f'Picking up goal block {block.visualization} at location {block.location}', self._agent_id)
        return GrabObject.__name__, {'object_id': block.obj_id}

    def drop_block(self) -> (str, {}):
        block = self._inventory
        self._inventory = None
        block.completed = True
        block.dropPoint.completed = True
        self._phase = Phase.WHAT_TO_DO

        self._sendMessage(f'Dropped goal block {block.visualization} at drop location {self._destination}', self._agent_id)
        return DropObject.__name__, {'object_id': block.obj_id}

    def open_door(self) -> (str, {}):
        self._phase = Phase.MOVING

        self._sendMessage(f"Opening door of {self._current_door_id.split('_')[0]}_{self._current_door_id.split('_')[1]}", self._agent_id)
        return OpenDoorAction.__name__, {'object_id': self._current_door_id}

    def plan_room_explore(self) -> (str, {}):

        assert 'room' in self._dest_id
        roomSquares = self._world.getRoom(self._dest_id).roomSquares
        self._navigator.reset_full()
        self._navigator.add_waypoints(roomSquares)

        self._sendMessage(f"Searching through {self._dest_id}", self._agent_id)
        return self.next(Phase.EXPLORE_ROOM)

    def explore_room(self) -> (str, {}):
        for key in self._current_state.keys():
            if 'Block_in' in key and key not in self._world.blocks:
                block_obj = self._current_state[key]
                block = MyBlock(block_obj, self._dest_id)
                self._world.addBlock(block)
                if block.isGoal:
                    self._sendMessage(f'Found goal block {block.visualization} at location {block.location}', self._agent_id)

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
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)

        self._location = self._current_state.get_self()['location']

        if self._phase != Phase.INITIALIZE:
            for room in state.get_all_room_names():
                if 'room' not in room:
                    continue
                door = state.get_room_doors(room)[0]
                self._world.getRoom(room).doorOpen = door['is_open']

        self.what_am_i_doing()

        return self.next(self._phase)

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
                    trustBeliefs[member] -= 0.1
                    break
        return trustBeliefs

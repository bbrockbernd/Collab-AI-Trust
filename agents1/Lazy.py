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
    EXPLORE_ROOM = 7


class Mode(enum.Enum):
    EXPLORING = 0
    GOAL = 1

class MyDropPoint:
    def __init__(self, shape: int, color: str, location: (int, int)):
        self.shape = shape
        self.location = location
        self.color = color

        # Not sure if this is unique
        self.myid = str(shape) + color

class MyGoal:
    def __init__(self, shape: int, color: str, location: (int, int)):
        self.shape = shape
        self.color = color
        self.location = location
        self.dropPoint = None
        self.x = location[0]
        self.y = location[1]
        self.completed = False
        self.myid = str(shape) + color

    def set_drop_point(self, dp: MyDropPoint):
        self.dropPoint = dp


class MyBlock:
    def __init__(self, shape: int, color: str, location: (int, int),
        isGoal=False):
        self.shape = shape
        self.color = color
        self.isGoal = isGoal
        self.location = location
        self.x = location[0]
        self.y = location[1]

    def setGoal(self, isGoal: bool):
        self.isGoal = isGoal


class MyRoom:

    def __init__(self, roomName: str, doorDict: {}):
        self.name = roomName
        self.contents = []
        self.goals = []
        self.explored = False
        self.doorLoc = doorDict['location']
        self.doorOpen = False
        self.doorId = doorDict['obj_id']


class MyWorld:
    def __init__(self):
        self.rooms = {}
        self.goals: [MyGoal] = []
        self.dropPoints: [MyDropPoint] = {}
        self.doorLocs = {}

    def addRoom(self, roomName: str, doorDict: {}):
        room = MyRoom(roomName, doorDict)
        self.rooms[room.name] = room
        self.doorLocs[room.doorLoc] = room.name

    def getRoom(self, name: str) -> MyRoom:
        return self.rooms[name]

    def getUnexploredRooms(self) -> List[MyRoom]:
        return [room for room in self.rooms.values() if not room.explored]

    def addGoal(self, goal: MyGoal):
        try:
            dp = self.dropPoints[goal.myid]
            goal.dropPoint = dp
            self.goals.append(goal)
        except KeyError:
            return

    def getGoals(self) -> [MyGoal]:
        return self.goals

    def hasGoals(self) -> bool:
        return len(self.goals) > 0

    def addDropPoint(self, dp: MyDropPoint):
        self.dropPoints[dp.myid] = dp

    def validateGoal(self, block: MyBlock) -> bool:
        for goal in self.goals:
            if goal.x == block.x and goal.y == block.y:
                return True
        return False

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

        self.doStuff = {
            Phase.INITIALIZE: self.init_vars,
            Phase.WHAT_TO_DO: self.what_to_do,
            Phase.CALCULATING: self.calculate,
            Phase.MOVING: self.moving,
            Phase.OPEN_DOOR: self.open_door,
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
        pass

    def init_vars(self) -> (str, {}):
        state = self._current_state
        for room in state.get_all_room_names():
            if 'room' not in room:
                continue
            self._world.addRoom(room, state.get_room_doors(room)[0])
        for key in state.keys():
            if "Collect_Block" in key:
                location = state[key]['location']
                colour = state[key]['visualization']['colour']
                shape = state[key]['visualization']['shape']
                self._world.addDropPoint(MyDropPoint(shape, colour, location))

        return self.next(Phase.WHAT_TO_DO)

    def what_to_do(self) -> (str, {}):
        if self._world.hasGoals():
            self._mode = Mode.GOAL
            self._destination = self._world.getGoals()[0].location
            self._dest_id = self._world.getGoals()[0]
            print("Going to Goal")
        else:
            self._mode = Mode.EXPLORING
            self._destination = self._world.getUnexploredRooms()[0].doorLoc
            self._destination = (self._destination[0], self._destination[1] + 1)
            print("Going to explore")

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
            return

        if action != None:
            return action, {}

        if self._mode == Mode.EXPLORING:
            # explore room
            return next(Phase.EXPLORE_ROOM)

    def open_door(self) -> (str, {}):
        self._phase = Phase.MOVING
        return OpenDoorAction.__name__, {'object_id': self._current_door_id}

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

import csv
import json
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
    PLAN_PATH_TO_CLOSED_DOOR = 1,
    FOLLOW_PATH_TO_CLOSED_DOOR = 2,
    OPEN_DOOR = 3


class MessageType(enum.Enum):
    MOVING = 1
    OPENING = 2
    SEARCHING = 3
    FOUND = 4
    PICKING_UP = 5
    DROPPED = 6
    FOUND_CONFIRMATION = 7
    TRUST_BELIEF = 8
    INVALID = 9


class BaseLineAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []
        self._log = {}  # Memory of recent actions by other agents
        self._actionHistory = {MessageType.PICKING_UP: [],
                               MessageType.DROPPED: []}  # History of Dropped and Picked Up blocks
        self._trustBeliefs = {}  # Trust Beliefs

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):
        return state

    def decide_on_bw4t_action(self, state: State):
        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
        # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBelief(agent_name, self._teamMembers, receivedMessages, state)

        while True:
            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()
                closedDoors = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door[
                        'is_open']]
                if len(closedDoors) == 0:
                    return None, {}
                # Randomly pick a closed door
                self._door = random.choice(closedDoors)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage('Moving to ' + self._door['room_name'], agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                # Open door
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

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
        self.received_messages = []  # Clear previous messages
        return receivedMessages

    '''
    Find the room (if any) based on a given location
    '''

    def _getRoom(self, location, state: State):
        rooms = state.get_all_room_names()
        for room in rooms:
            objects = state.get_room_objects(room)
            for obj in objects:
                if obj['location'] == location:
                    return room
        return ''

    '''
    Compute the trust belief value based on trust and reputation for all given agents
    Direct Experiences influence more than Indirect experience and Reputation
    '''

    def _computeTrustBeliefs(self, agents):
        trust_beliefs = {}
        for [agent, direct_exp, indirect_exp, rep] in agents:
            trust_beliefs[agent] = self._computeTrustBelief(direct_exp, indirect_exp, rep)
        return trust_beliefs

    '''
    Compute the trust belief value based on trust and reputation
    Direct Experiences influence more than Indirect experience and Reputation
    '''

    def _computeTrustBelief(self, direct_exp, indirect_exp, rep):
        return (3 * float(direct_exp) + float(indirect_exp) + float(rep)) / 5

    '''
    Returns True if an agent can be trusted
    '''

    def _trustInAgent(self, agent_id: str) -> bool:
        if self._trustBeliefs[agent_id] < 0:
            return False
        else:
            return True

    '''
    Has to be overwritten by child classes
    '''

    def _validateBlock(self, location: (int, int), color: str, shape: int) -> int:
        return 0

    '''
    Returns the history of data received from PICKING_UP and DROPPED messages
    '''

    def _blockActions(self, action: MessageType) -> List[List]:
        if action == MessageType.DROPPED:
            return self._actionHistory[MessageType.DROPPED]
        elif action == MessageType.PICKING_UP:
            return self._actionHistory[MessageType.PICKING_UP]
        else:
            return []

    '''
    Transform text ending with "location (x, y)" to a tupple (x,y)
    '''

    def _getLocationFromMessage(self, message):
        loc = message[message.find("location ("):-1]
        return (int(loc[loc.find("(") + 1:loc.find(',')]), int(loc[loc.find(",") + 2:]))

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
        received = received.replace('T', 't').replace('F', 'f')
        message = received.split()
        try:
            if len(message) == 3 and ' '.join(message[:2]) == 'Moving to':
                return MessageType.MOVING, message[2]
            elif len(message) == 4 and ' '.join(message[:3]) == 'Opening door of':
                return MessageType.OPENING, message[3]
            elif len(message) == 3 and ' '.join(message[:2]) == 'Searching through':
                return MessageType.SEARCHING, message[2]
            elif len(message) > 3 and ' '.join(message[:3]) == 'found goal block':
                return MessageType.FOUND, [
                    json.loads(received[received.find('{'):received.find('}') + 1].replace("'", '"')),
                    self._getLocationFromMessage(received)]
            elif len(message) > 3 and ' '.join(message[:3]) == 'Dropped goal block':
                return MessageType.DROPPED, [
                    json.loads(received[received.find('{'):received.find('}') + 1].replace("'", '"')),
                    self._getLocationFromMessage(received)]
            elif len(message) > 4 and ' '.join(message[:4]) == 'Picking up goal block':
                return MessageType.PICKING_UP, [
                    json.loads(received[received.find('{'):received.find('}') + 1].replace("'", '"')),
                    self._getLocationFromMessage(received)]
            elif len(message) == 5 and ' '.join(message[:3]) == 'found block by':
                return MessageType.FOUND_CONFIRMATION, [message[3], message[4]]
            elif len(message) == 6 and ' '.join(message[:3]) == 'trust belief of':
                return MessageType.TRUST_BELIEF, [message[3], float(message[5])]
            else:
                return MessageType.INVALID, []
        except Exception:
            print("ERROR WITH PREPROCESSING FOLLOWING MESSAGE: " + received)
            return MessageType.INVALID, []

    '''
    Verify that the same message is not processed twice
    '''

    def _checkIfMessageAlreadyRecieved(self, member, message_type, message_data):
        if member in self._log:
            if message_type in self._log[member]:
                if message_data == self._log[member][message_type]:
                    return True
        return False

    '''
    Trust mechanism (same) for all Agents
    '''

    def _trustBelief(self, name, members, received, state: State):
        # Read (or initialize) memory file
        default = 0.0
        truth_reward = 0.1
        lie_cost = 0.4
        filename = name + '_memory.csv'
        params = ['Agent', 'Direct Experiences', 'Indirect Experiences', 'Reputation']
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
                    agents.append([member, default, default, default])
                memory.writerows(agents)

        agents = [[agent[0], float(agent[1]), float(agent[2]), float(agent[3])] for agent in agents]
        # Process received messages
        for member in received.keys():
            if member == name:
                print("PROCESSING OWN MESSAGE!")
                continue
            # Agent index for trust modification
            member_index = [name for name, direct, indirect, reputation in agents].index(member)

            for message in received[member]:
                message_type, message_data = self._normalizeMessage(message)  # Preprocess message
                if not self._checkIfMessageAlreadyRecieved(member, message_type, message_data):  # Check for duplicates
                    if member in self._log:
                        # Check if found previous instance of message type
                        type_already_exists = message_type in self._log[member]

                        # Check if message is of type FOUND and already has data
                        if message_type == MessageType.FOUND:

                            # Store message
                            if type_already_exists:
                                found_blocks = self._log[member][message_type]
                                found_blocks.append(message_data)
                                self._log[member][message_type] = found_blocks
                            else:
                                self._log[member][message_type] = [message_data]

                            # Trust: Check if blocks FOUND are in the room the agent said they were SEARCHING
                            room = self._getRoom(message_data[1], state)
                            if room != '':
                                if MessageType.SEARCHING in self._log[member] and self._log[member][
                                    MessageType.SEARCHING] \
                                        == room:
                                    agents[member_index][1] += truth_reward
                                else:
                                    agents[member_index][1] -= lie_cost

                            # Trust: Check if FOUND block by other matches what you know
                            block_confirmation = self._validateBlock(message_data[1],
                                                                     message_data[0]['colour'],
                                                                     message_data[0]['shape'])

                            if block_confirmation == 1:
                                agents[member_index][1] += truth_reward
                                self._sendMessage("Found block by " + member + " approved", name)

                            elif block_confirmation == -1:
                                agents[member_index][1] -= lie_cost
                                self._sendMessage("Found block by " + member + " denied", name)

                        # Check if message is of type PICKING_UP or DROPPED and already has data (max 2)
                        elif message_type == MessageType.PICKING_UP or message_type == MessageType.DROPPED:

                            # Store data in history of actions List[List[str, tuple[int, int], str]]
                            history = self._actionHistory[message_type]
                            data = [message_data[0], message_data[1], member]
                            history.append(data)
                            self._actionHistory[message_type] = history

                            # Store message in log
                            if type_already_exists:
                                if len(self._log[member][message_type]) > 1:  # Strong can pick up 2
                                    self._log[member][message_type] = [message_data]
                                else:
                                    blocks = self._log[member][message_type]
                                    blocks.append(message_data)
                                    self._log[member][message_type] = blocks
                            else:
                                self._log[member][message_type] = [message_data]

                            # Trust: For PICKING_UP check if agent moved to that room beforehand
                            if message_type == MessageType.PICKING_UP:
                                room = self._getRoom(message_data[1], state)
                                if room != '':
                                    if MessageType.MOVING in self._log[member] and \
                                            self._log[member][MessageType.MOVING] == room:
                                        agents[member_index][1] += truth_reward
                                    else:
                                        agents[member_index][1] -= lie_cost

                                # Reputation Broadcast
                                trust_b = self._computeTrustBelief(agents[member_index][1], agents[member_index][2],
                                                                   agents[member_index][3])
                                self._sendMessage('Trust belief of ' + member + ' : ' + str(trust_b), name)

                            # Trust: For DROPPED check if agent picked up that block before
                            if message_type == MessageType.DROPPED:
                                if MessageType.PICKING_UP in self._log[member] and message_data[0] in \
                                        [block for block, location in self._log[member][MessageType.PICKING_UP]]:
                                    agents[member_index][1] += truth_reward
                                else:
                                    agents[member_index][1] -= lie_cost

                        # Indirect Experience
                        elif message_type == MessageType.FOUND_CONFIRMATION:
                            try:
                                agent_index = [name for name, direct, indirect, reputation in agents].index(
                                    message_data[0])
                                if message_data[1] == 'approved':
                                    agents[agent_index][2] += truth_reward
                                elif message_data[1] == 'denied':
                                    agents[agent_index][2] -= lie_cost
                            except ValueError:  # This happens if message is about itself
                                pass

                        elif message_type == MessageType.TRUST_BELIEF:
                            try:
                                agent_index = [name for name, direct, indirect, reputation in agents].index(
                                    message_data[0])
                                agents[agent_index][3] = (agents[agent_index][3] + message_data[1]) / 2
                            except ValueError:  # This happens if message is about itself
                                pass

                        # All the other messages have max 1 consecutive type of message
                        else:
                            # Store message
                            self._log[member][message_type] = message_data

                            # Trust: For OPENING check if correct door is indeed open
                            if message_type == MessageType.OPENING:
                                open_doors_room = [door['room_name'] for door in state.values()
                                                   if 'class_inheritance' in door and 'Door' in door[
                                                       'class_inheritance'] and door['is_open']]
                                if message_data in open_doors_room:
                                    agents[member_index][1] += truth_reward
                                else:
                                    agents[member_index][1] -= lie_cost

                            # Trust: For SEARCHING check if agent was moving to that room
                            if message_type == MessageType.SEARCHING:
                                if MessageType.MOVING in self._log[member] and self._log[member][MessageType.MOVING] \
                                        == message_data:
                                    agents[member_index][1] += truth_reward
                                else:
                                    agents[member_index][1] -= lie_cost
                    else:
                        self._log[member] = {}
                        self._log[member][message_type] = message_data

        # Save back to memory file
        with open(filename, 'w', newline='') as mem_file:
            memory = csv.writer(mem_file)
            memory.writerow(params)
            memory.writerows(agents)

        self._trustBeliefs = self._computeTrustBeliefs(agents)

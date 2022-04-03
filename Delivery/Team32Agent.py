import copy
import warnings
import numpy as np
import enum, random
import random
import ast
import enum, random

from typing import List, Dict
from agents1.BW4TBaselineAgent import BaseLineAgent
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.agents.agent_brain import AgentBrain
from matrx.actions import GrabObject, RemoveObject, OpenDoorAction, CloseDoorAction, DropObject
from matrx.agents.agent_utils.state import State
from matrx.messages import Message

#GROUP 32


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
    
class Phase(enum.Enum):
    # For lazy
    INITIALIZE = 0
    WHAT_TO_DO = 1
    CALCULATING = 2
    MOVING = 3
    OPEN_DOOR = 4
    PICKUP = 5
    DROP = 6
    PLAN_ROOM_EXPLORE = 7
    EXPLORE_ROOM = 8
    
    # For liar, strong and colorblind
    SET_UP_VARIABLES = 0,
    PLAN_PATH_TO_UNSEARCHED_ROOM = 1,
    FOLLOW_PATH_TO_UNSEARCHED_ROOM = 2,
    PLAN_ROOM_EXPLORATION = 3,
    PLAN_TO_GOAL_BLOCK = 6,
    FOLLOW_PATH_TO_GOAL_BLOCK=7,
    GRAB_BLOCK = 18,
    PLAN_TO_DROP_ZONE = 9,
    FOLLOW_PATH_TO_DROP_ZONE = 10,
    PLAN_PATH_TO_VERIFY_COLLECTION = 11,
    FOLLOW_PATH_TO_VERIFY_COLLECTION = 12,
    PLAN_PATH_TO_REMOVE_ALL_BLOCKS = 13,
    FOLLOW_PATH_TO_REMOVE_ALL_BLOCKS = 14,
    REMOVE_ALL_BLOCKS = 15,
    REPLACE_ALL_BLOCKS = 15,
    DOUBLE_DROPPOFF = 17

    
class BW4TAgentBrain(AgentBrain):
    """ An artificial agent whose behaviour can be programmed to be, for example, (semi-)autonomous.
    This brain inherits from the normal MATRX AgentBrain but with one small adjustment in the function '_set_messages' making it possible to identify the sender of messages.
    """


    def __init__(self,memorize_for_ticks=None):
        """ Defines the behavior of an agent.
        This class is the place where all the decision logic of an agent is
        contained. This class together with the
        :class:`matrx.objects.agent_body.AgentBody` class represent a full agent.
        This agent brain simply selects a random action from the possible actions
        it can do.
        When you wish to create a new agent, this is the class you need
        to extend. In specific these are the functions you should override:
        * :meth:`matrx.agents.agent_brain.initialize`
            Called before a world starts running. Can be used to initialize
            variables that can only be initialized after the brain is connected to
            its body (which is done by the world).
        * :meth:`matrx.agents.agent_brain.filter_observations`
            Called before deciding on an action to allow detailed and agent
            specific filtering of the received world state.
        * :meth:`matrx.agents.agent_brain.decide_on_action`
            Called to decide on an action.
        * :meth:`matrx.agents.agent_brain.get_log_data`
            Called by data loggers to obtain data that should be logged from this
            agent internal reasoning.
        Attributes
        ----------
        action_set: [str, ...]
            List of actions this agent can perform.
        agent_id: str
            The unique identified of this agent's body in the world.
        agent_name: str
            The name of this agent.
        agent_properties: dict
            A dictionary of this agent's
            :class:`matrx.objects.agent_body.AgentBody` properties. With as keys
            the property name, and as value the property's value.
            These can be adjusted iff they are said to be adjustable (e.g. inside
            the attribute `keys_of_agent_writable_props`).
        keys_of_agent_writable_props: [str, ...]
            List of property names that this agent can adjust.
        messages_to_send: [Message, ...]
            List of messages this agent will send. Use the method
            :meth:`matrx.agents.agent_brain.AgentBrain.send_message` to append to
            this list.
        previous_action: str
            The name of the previous performed or attempted action.
        previous_action_result: ActionResult
            The :class:`matrx.actions.action.ActionResult` of the previously
            performed or attempted action.
        received_messages: [Message, ...]
            The list of received messages.
        rnd_gen: Random
            The random generator for this agent.
        rnd_seed: int
            The random seed with which this agent's `rnd_gen` was initialized. This
            seed is based on the master random seed given of the
            :class:`matrx.grid_world.GridWorld`.
        """
        # Class variables for tracking the past action and its result
        self.previous_action = None
        self.previous_action_result = None

        # A list of messages that may be filled by this agent, which is retrieved by the GridWorld and send towards the
        # appropriate agents.
        self.messages_to_send = []
        self.received_messages = []

        # Filled by the WorldFactory during self.factory_initialise()
        self.agent_id = None
        self.agent_name = None
        self.action_set = None  # list of action names (strings)
        self.sense_capability = None
        self.rnd_gen = None
        self.rnd_seed = None
        self.agent_properties = {}
        self.keys_of_agent_writable_props = []
        self.__memorize_for_ticks = memorize_for_ticks

        # The central state property (an extended dict with unique searching capabilities)
        self._state = None

    def initialize(self):
        """ Method called by any world when it starts.
        When adding an agent to a :class:`matrx.grid_world.GridWorld`, through
        a world builer, you only pass the class of your agent brain, not the
        actual instance. Instead, this instance is made by the builder when
        a new world is created and ran. At that point this method is called.
        That makes this method the ideal place for any initialization or
        reset you want your agent brain to do when starting a world or between
        worlds.
        Importantly, this method is called after the builder assigned things
        to it such as its location, name and object ID. As this method is
        called afterwards, it allows you to do things related to to those
        properties.
        An example is when you run the same world multiple times. In that case
        the instance of your agent brain will have attributes with values from
        the previous run. This method can be used to reset them.  
        """
        self.previous_action = None
        self.previous_action_result = None
        self.messages_to_send = []
        self.received_messages = []
        self._init_state()

    def filter_observations(self, state):
        """ Filters the world state before deciding on an action.
        In this method you filter the received world state to only those
        properties and objects the agent is actually supposed to see.
        Currently the world returns ALL properties of ALL objects within a
        certain range(s), as specified by
        :class:`matrx.agents.capabilities.capability.SenseCapability`. But
        perhaps some objects are obscured because they are behind walls and
        this agent is not supposed to look through walls, or an agent is not
        able to see some properties of certain objects (e.g. colour).
        The adjusted world state that this function returns is directly fed to
        the agent's decide function. Furthermore, this returned world state is
        also fed through the MATRX API to any visualisations.
        Override this method when creating a new AgentBrain and you need to
        filter the world state further.
        Parameters
        ----------
        state : State
            A state description containing all perceived
            :class:`matrx.objects.env_object.EnvObject` and objects inheriting
            from this class within a certain range as defined by the
            :class:`matrx.agents.capabilities.capability.SenseCapability`.
            The keys are the unique identifiers, as values the properties of
            an object. See :class:`matrx.objects.env_object.EnvObject` for the
            kind of properties that are always included. It will also contain
            all properties for more specific objects that inherit from that
            class.
            Also includes a 'world' key that describes common information about
            the world (e.g. its size).
        Returns
        -------
        filtered_state : State
            A dictionary similar to `state` but describing the filtered state
            this agent perceives of the world.
        Notes
        -----
        A future version of MATRX will include handy utility function to make
        state filtering less of a hassle (e.g. to easily remove specific
        objects or properties, but also ray casting to remove objects behind
        other objects)
        """

        return state

    def decide_on_action(self, state):
        """ Contains the decision logic of the agent.
        This method determines what action the agent should perform. The
        :class:`matrx.grid_world.GridWorld` is responsible for deciding when
        an agent can perform an action, if so this method is called for each
        agent and fed with the world state from the `filter_observations`
        method.
        Two things need to be determined: action name and action arguments.
        The action is returned simply as the class name (as a string), and the
        action arguments as a dictionary with the keys the names of the keyword
        arguments. See the documentation of that action to find out which
        arguments.
        An argument that is always possible is that of action_duration, which
        denotes how many ticks this action should take and overrides the
        action duration set by the action implementation.
        Parameters
        ----------
        state : State
            A state description as given by the agent's
            :meth:`matrx.agents.agent_brain.AgentBrain.filter_observations` method.
        Returns
        -------
        action_name : str
            A string of the class name of an action that is also in the
            `action_set` class attribute. To ensure backwards compatibility
            we advise to use Action.__name__ where Action is the intended
            action.
        action_args : dict
            A dictionary with keys any action arguments and as values the
            actual argument values. If a required argument is missing an
            exception is raised, if an argument that is not used by that
            action a warning is printed. The argument applicable to all action
            is `action_duration`, which sets the number ticks the agent is put
            on hold by the GridWorld until the action's world mutation is
            actual performed and the agent can perform a new action (a value of
            0 is no wait, 1 means to wait 1 tick, etc.).
        Notes
        -----
        A future version of MATRX will include handy utility function to make
        agent decision-making less of a hassle. Think of a
        Belief-Desire-Intention (BDI) like structure, and perhaps even support
        for learning agents.
        """

        # send a random message once in a while
        if self.rnd_gen.random() < 0.1:
            # Get all agents in our state.
            # The codeline below can return three things:
            # - None                    -> no agents found (impossible as we are in an agent right now)
            # - an agent object         -> only a single agent found
            # - a list of agent objects -> multiple agents found
            # Also see for state usage:
            # https://github.com/matrx-software/matrx/blob/master/matrx/cases/bw4t/bw4t_agents.py
            agents = state[{"isAgent": True}]

            # If we found multiple agents, randomly select the ID of one of them or otherwise the ID of the only agent
            to_id = self.rnd_gen.choice(agents)['obj_id'] if isinstance(agents, list) else agents['obj_id']

            self.send_message(Message(content=f"Hello, my name is (agent) {self.agent_name} and I sent this message at "
                                              f"tick {state['World']['nr_ticks']}",
                                      from_id=self.agent_id,
                                      to_id=to_id))
        # Select a random action
        if self.action_set:
            action = self.rnd_gen.choice(self.action_set)
        else:
            action = None

        action_kwargs = {}

        if action == RemoveObject.__name__:
            action_kwargs['object_id'] = None

            # Get all perceived objects
            objects = list(state.keys())
            # Remove yourself from the object id list
            objects.remove(self.agent_properties["obj_id"])
            # Remove all objects that have 'agent' in the name (so we do not remove those, though agents without agent
            # in their name can still be removed).
            objects = [obj for obj in objects if 'agent' not in obj]
            # Choose a random object id (safety for when it is empty)
            if objects:
                object_id = self.rnd_gen.choice(objects)
                # Assign it
                action_kwargs['object_id'] = object_id
                # Select range as just enough to remove that object
                remove_range = int(np.ceil(np.linalg.norm(
                    np.array(state[object_id]['location']) - np.array(
                        state[self.agent_properties["obj_id"]]['location']))))
                # Safety for if object and agent are in the same location
                remove_range = max(remove_range, 0)
                # Assign it to the arguments list
                action_kwargs['remove_range'] = remove_range
            else:
                action_kwargs['object_id'] = None
                action_kwargs['remove_range'] = 0

        # if the agent randomly chose a grab action, choose a random object to pickup
        elif action == GrabObject.__name__:
            # Set grab range
            grab_range = 1

            # Set max amount of objects
            max_objects = 3

            # Assign it to the arguments list
            action_kwargs['grab_range'] = grab_range
            action_kwargs['max_objects'] = max_objects

            # Get all perceived objects
            objects = list(state.keys())

            # Remove yourself from the object id list
            objects.remove(self.agent_properties["obj_id"])
            # Remove all objects that have 'agent' in the name (so we do not remove those, though agents without agent
            # in their name can still be removed).
            objects = [obj for obj in objects if 'agent' not in obj]
            # Choose a random object id (safety for when it is empty)

            object_in_range = []
            for object_id in objects:
                # Select range as just enough to grab that object
                dist = int(np.ceil(np.linalg.norm(
                    np.array(state[object_id]['location']) - np.array(
                        state[self.agent_properties["obj_id"]]['location']))))
                if dist <= grab_range and state[object_id]["is_movable"]:
                    object_in_range.append(object_id)

            if object_in_range:
                # Select object
                object_id = self.rnd_gen.choice(object_in_range)

                # Assign it
                action_kwargs['object_id'] = object_id
            else:
                action_kwargs['object_id'] = None

        # if we randomly chose to do a open or close door action, find a door to open/close
        elif action == OpenDoorAction.__name__ or action == CloseDoorAction.__name__:

            action_kwargs['door_range'] = 1  # np.inf
            action_kwargs['object_id'] = None

            # Get all doors from the perceived objects
            objects = list(state.keys())
            doors = [obj for obj in objects
                     if 'class_inheritance' in state[obj] and state[obj]['class_inheritance'][0] == "Door"]

            # choose a random door
            if len(doors) > 0:
                action_kwargs['object_id'] = self.rnd_gen.choice(doors)

        return action, action_kwargs

    def get_log_data(self):
        """ Provides a dictionary of data for any Logger
        This method functions to relay data from an agent's decision logic (this AgentBrain class) through the GridWorld
        into a Logger. Here it can be further processed and stored.
        Returns
        -------
        data : dict
            A dictionary with keys identifiable names and the data as its value.
        """
        return {}

    def send_message(self, message):
        """  Sends a Message from this agent to others
        Method that allows you to construct a message that will be send to either a specified agent, a team of agents
        or all agents.
        Parameters
        ----------
        message : Message
            A message object that needs to be send. Should be of type Message. It's to_id can contain a single
            recipient, a list of recipients or None. If None, it is send to all other agents.
        """
        # Check if the message is a true message
        self.__check_message(message, self.agent_id)
        # Add the message to our list
        self.messages_to_send.append(message)

    def is_action_possible(self, action, action_kwargs):
        """ Checks if an action would be possible.
        This method can be called from the AgentBrain to check if a certain action is possible to perform with the
        current state of the GridWorld. It requires as input an action name and its arguments (if any), same as the
        decide_on_action method should return.
        This method does not guarantees that if the action is return by the brain it actually succeeds, as other agents
        may intervene.
        Parameters
        ----------
        action : str
            The name of an Action class.
        action_kwargs : dict
            A dictionary with keys any action arguments and as values the actual argument values.
        Returns
        -------
        succeeded : bool
            True if the action can be performed, False otherwise.
        action_results : ActionResult
            An ActionResult object containing the success or failure of the action, and (if failed) the reason why.
        """
        action_result = self.__callback_is_action_possible(self.agent_id, action, action_kwargs)

        return action_result.succeeded, action_result

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):

        # Check if the return filtered state is a differently created State
        # object, if so, raise the warning that we are overwriting it.
        if new_state is not self.state:
            warnings.warn(f"Overwriting State object of {self.agent_id}. This "
                          f"will cause any stored memory to be gone for good "
                          f"as this was stored in the previous State object.")

        if isinstance(new_state, dict):
            raise TypeError(f"The new state should of type State, is of "
                            f"type {new_state.__class__}")

        self._state = new_state

    @property
    def memorize_for_ticks(self):
        return self.__memorize_for_ticks

    def create_context_menu_for_other(self, agent_id_who_clicked, clicked_object_id, click_location):
        """ Generate options for a context menu for a specific object/location that a user NOT controlling this
        human agent opened.
        Thus: another human agent selected this agent, opened a context menu by right clicking on an object or location.
        This function is called. It should return actions, messages, or other info for what this agent can do relevant
        to that object / location. E.g. pick it up, move to it, display information on it, etc.
        Example usecase: tasking another agent that is not yourself, e.g. to move to a specific location.
        For the default MATRX visualization, the context menu is opened by right clicking on an object. This function
        should generate a list of options (actions, messages, or something else) which relate to that object or location.
        Each option is in the shape of a text shown in the context menu, and a message which is send to this agent if
        the user actually clicks that context menu option.
        Parameters
        ----------
        agent_id_who_clicked : str
            The ID of the (human) agent that selected this agent and requested for a context menu.
        clicked_object_id : str
            A string indicating the ID of an object. Is None if the user clicked on a background tile (which has no ID).
        click_location : list
            A list containing the [x,y] coordinates of the object on which the user right clicked.
        Returns
        -------
         context_menu : list
            A list containing context menu items. Each context menu item is a dict with a 'OptionText' key, which is
            the text shown in the menu for the option, and a 'Message' key, which is the message instance that is sent
            to this agent when the user clicks on the context menu option.
        """
        print("Context menu other")
        context_menu = []

        # Generate a context menu option for every action
        for action in self.action_set:
            context_menu.append({
                "OptionText": f"Do action: {action}",
                "Message": Message(content=action, from_id=clicked_object_id, to_id=self.agent_id)
            })
        return context_menu

    def _factory_initialise(self, agent_name, agent_id, action_set, sense_capability, agent_properties,
                            customizable_properties, rnd_seed, callback_is_action_possible):
        """ Private MATRX function.
        Initialization of the brain by the WorldBuilder.
        Called by the WorldFactory to initialise this agent with all required properties in addition with any custom
        properties. This also sets the random number generator with a seed generated based on the random seed of the
        world that is generated.
        Parameters
        ----------
        agent_name : str
            The name of the agent.
        agent_id : str
            The unique ID given by the world to this agent's avatar. So the agent knows what body is his.
        action_set : str
            The list of action names this agent is allowed to perform.
        sense_capability : SenseCapability
            The SenseCapability of the agent denoting what it can see withing what range.
        agent_properties : dict
            The dictionary of properties containing all mandatory and custom properties.
        customizable_properties : list
            A list of keys in agent_properties that this agent is allowed to change.
        rnd_seed : int
            The random seed used to set the random number generator self.rng
        callback_is_action_possible : callable
            A callback to a GridWorld method that can check if an action is possible.
        """

        # The name of the agent with which it is also known in the world
        self.agent_name = agent_name

        # The id of the agent
        self.agent_id = agent_id

        # The names of the actions this agent is allowed to perform
        self.action_set = action_set

        # Setting the random seed and rng
        self.rnd_seed = rnd_seed
        self._set_rnd_seed(seed=rnd_seed)

        # Initializing the State object
        self._init_state()

        # The SenseCapability of the agent; what it can see and within what range
        self.sense_capability = sense_capability

        # Contains the agent_properties
        self.agent_properties = agent_properties

        # Specifies the keys of properties in self.agent_properties which can  be changed by this Agent in this file. If
        # it is not writable, it can only be  updated through performing an action which updates that property (done by
        # the environment).
        # NOTE: Changing which properties are writable cannot be done during runtime! Only when adding it to the world.
        self.keys_of_agent_writable_props = customizable_properties

        # A callback to the GridWorld instance that can check whether any action (with its arguments) will succeed and
        # if not why not (in the form of an ActionResult).
        self.__callback_is_action_possible = callback_is_action_possible

    def _get_action(self, state, agent_properties, agent_id):
        """ Private MATRX function
        The function the environment calls. The environment receives this function object and calls it when it is time
        for this agent to select an action.
        Note; This method should NOT be overridden!
        Parameters
        ----------
        state_dict: dict
            A state description containing all properties of EnvObject that are within a certain range as defined by
            self.sense_capability. It is a list of properties in a dictionary
        agent_properties: dict
            The properties of the agent, which might have been changed by the environment as a result of actions of
            this or other agents.
        agent_id: str
            the ID of this agent
        Returns
        -------
         filtered_state : dict
            The filtered state of this agent
        agent_properties : dict
            the agent properties which the agent might have changed,
        action : str
            an action string, which is the class name of one of the actions in the Action package.
        action_kwargs : dict
            Keyword arguments for the action
        """
        # Process any properties of this agent which were updated in the environment as a result of actions
        self.agent_properties = agent_properties

        # Update the state property of an agent with the GridWorld's state dictionary
        self.state.state_update(state.as_dict())

        # Call the filter method to filter the observation
        self.state = self.filter_observations(self.state)

        # Call the method that decides on an action
        action, action_kwargs = self.decide_on_action(self.state)

        # Store the action so in the next call the agent still knows what it did
        self.previous_action = action

        # Return the filtered state, the (updated) properties, the intended actions and any keyword arguments for that
        # action if needed.
        return self.state, self.agent_properties, action, action_kwargs

    def _fetch_state(self, state):
        self.state.state_update(state.as_dict())
        filtered_state = self.filter_observations(self.state)
        return filtered_state

    def _get_log_data(self):
        return self.get_log_data()

    def _set_action_result(self, action_result):
        """ A function that the environment calls (similarly as the self.get_action method) to set the action_result of the
        action this agent decided upon.
        Note, that the result is given AFTER the action is performed (if possible).
        Hence it is named the self.previous_action_result, as we can read its contents when we should decide on our
        NEXT action after the action whose result this is.
        Note; This method should NOT be overridden!
        Parameters
        ----------
        action_result : ActionResult
            An object that inherits from ActionResult, containing a boolean whether the action succeeded and a string
            denoting the reason why it failed (if it did so).
        """
        self.previous_action_result = action_result

    def _set_rnd_seed(self, seed):
        """ The function that seeds this agent's random seed.
        Note; This method should NOT be overridden!
        Parameters
        ----------
        seed : int
            The random seed this agent needs to be seeded with.
        """
        self.rnd_seed = seed
        self.rnd_gen = np.random.RandomState(self.rnd_seed)

    def _get_messages(self, all_agent_ids):
        """ Retrieves all message objects the agent has made in a tick, and returns those to the GridWorld for sending.
        It then removes all these messages!
        This method is called by the GridWorld.
        Note; This method should NOT be overridden!
        Parameters
        ----------
        all_agent_ids
            IDs of all agents
        Returns
        -------
            A list of message objects with a generic content, the sender (this agent's id) and optionally a
            receiver.
        """
        # # preproccesses messages such that they can be understand by the gridworld
        # preprocessed_messages = self.preprocess_messages(this_agent_id=self.agent_id, agent_ids=all_agent_ids,
        # messages=self.messages_to_send)

        send_messages = copy.copy(self.messages_to_send)

        # Remove all messages that need to be send, as we have send them now
        self.messages_to_send = []

        return send_messages

    def _set_messages(self, messages=None):
        """
        This method is called by the GridWorld.
        It sets all messages intended for this agent to a list that it can access and read.
        Note; This method should NOT be overridden!
        Parameters
        ----------
        messages : Dict (optional, default, None)
            A list of dictionaries that contain a 'from_id', 'to_id' and 'content. If messages is set to None (or no
            messages are used as input), only the previous messages are removed
        """

        # We empty all received messages as this is from the previous tick
        # self.received_messages = []

        # Loop through all messages and create a Message object out of the dictionaries.
        for mssg in messages:

            # Check if the message is of type Message (its content contains the actual message)
            BW4TAgentBrain.__check_message(mssg, self.agent_id)

            # Since each message is secretly wrapped inside a Message (as its content), we unpack its content and
            # set that as the actual received message.
            received_message = mssg.content

            # Add the message object to the received messages
            self.received_messages.append(mssg)

    def _init_state(self):
        self._state = State(memorize_for_ticks=self.memorize_for_ticks,
                            own_id=self.agent_id)

    @staticmethod
    def __check_message(mssg, this_agent_id):
        if not isinstance(mssg, Message):
            raise Exception(f"A message to {this_agent_id} is not, nor inherits from, the class {Message.__name__}."
                            f" This is required for agents to be able to send and receive them.")







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
        self.size = block_obj['visualization']['size']
        self.isGoal = isGoal
        self.location = block_obj['location']
        self.dropPoint = None
        self.completed = False
        self.room = room
        self.myid = str(self.shape) + self.color
        self.obj_id = block_obj['obj_id']
        self.visualization = {
            "size": self.size,
            "shape": self.shape,
            "colour": self.color
        }


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
        self._isCarrying = False
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
            self._phase = Phase.WHAT_TO_DO

        if self._counter > 7 and self._quitting and self._phase is not Phase.EXPLORE_ROOM:
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
        else:
            self._mode = Mode.EXPLORING
            if len(self._world.getUnexploredRooms()) == 0:
                for room in self._world.rooms:
                    self._world.getRoom(room).explored = False
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
        if self._mode == Mode.GOAL and self._inventory is not None and not self._isCarrying:
            self._inventory = None
            return self.next(Phase.WHAT_TO_DO)

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

        self._isCarrying = len(state[self.agent_id]['is_carrying']) > 0

        if self._phase != Phase.INITIALIZE:
            for room in state.get_all_room_names():
                if 'room' not in room:
                    continue
                door = state.get_room_doors(room)[0]
                self._world.getRoom(room).doorOpen = door['is_open']

        for agent in receivedMessages.keys():
            for msg in receivedMessages[agent]:
                if not self._trustInAgent(agent):
                    continue
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

    def _validateBlock(self, location: (int, int), color: str, shape: int) -> int:
        if location not in self._checked_locations:
            return 0
        for block in self._world.blocks.values():
            if block.location == location and (block.color == color or color == "?") and block.shape == shape:
                return 1
        return -1









class Liar(BaseLineAgent):

    def __init__(self, settings:Dict[str,object]):
        super().__init__(settings)
        self._phase=Phase.SET_UP_VARIABLES
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
                self.updateBlocks(state)
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
                self.updateBlocks(state)
                self._phase=Phase.PLAN_ROOM_EXPLORATION                
                if not self._door['is_open']:
                    for block in state.keys():
                        if "door" in block and state[block]["obj_id"] == self._door['obj_id'] and not state[block]["is_open"]:
                            self._sendDoorOpenMessage(state)
                            return "OpenDoorAction" , {'object_id':self._door['obj_id']}  
            
            elif Phase.PLAN_ROOM_EXPLORATION==self._phase:
                self.updateBlocks(state)
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
                self.updateBlocks(state)
                possible = self._possibleToPlanPathToGoalBlock()
                if possible == False:
                    self._phase=Phase.PLAN_PATH_TO_VERIFY_COLLECTION
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
                self.updateBlocks(state)
                blocks = self.detectBlocksAround(state)
                ids = []
                for block in blocks:
                    ids.append(block['obj_id'])
                if self.blockToGrab['obj_id'] not in ids:
                    #BLOCK NOT ON LAST KNOWN LOCATION
                    del self.knownBlocks[self.blockToGrab['obj_id']]
                    self._phase=Phase.PLAN_PATH_TO_UNSEARCHED_ROOM
                    continue
                
                self._sendGrabBlockMessage(state)
                self._phase=Phase.PLAN_TO_DROP_ZONE
                return "GrabObject", {'object_id':self.blockToGrab['obj_id'] } 
            
            if Phase.PLAN_TO_DROP_ZONE==self._phase:
                self.updateBlocks(state)
                if(len(self.agent_properties['is_carrying']) == 0):
                    #NO BLOCKS BRAPPED
                    self._phase=Phase.PLAN_PATH_TO_UNSEARCHED_ROOM
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
                
                if self.checkAllCollectBlocksPresent():
                    self._phase=Phase.PLAN_PATH_TO_REMOVE_ALL_BLOCKS
                else:
                    self._phase=Phase.PLAN_PATH_TO_UNSEARCHED_ROOM

                    
            if Phase.PLAN_PATH_TO_REMOVE_ALL_BLOCKS==self._phase:
                self.updateBlocks(state)
                id = list(self.collectBlocks.keys())
                id.sort(reverse=True)
                (x,y) = self.collectBlocks[id[0]]['location']
                location = (x-1,y)
                
                self._navigator.reset_full()
                self._navigator.add_waypoints([location])
                
                self._phase=Phase.FOLLOW_PATH_TO_REMOVE_ALL_BLOCKS
                
            if Phase.FOLLOW_PATH_TO_REMOVE_ALL_BLOCKS==self._phase:
                self.updateBlocks(state)
                
                self._state_tracker.update(state)
                
                action = self._navigator.get_move_action(self._state_tracker)
                
                if action!=None:
                    return action, {}   
                if random.random() < 0.9:
                    return None, {}
                self._phase=Phase.REMOVE_ALL_BLOCKS
            
            if Phase.REMOVE_ALL_BLOCKS == self._phase:
                self.updateBlocks(state)
                (x,y) = state[self.agent_id]['location']
                location = (x+1,y)
                blocks = self.detectBlocksAround(state)

                carrying = self.agent_properties['is_carrying']
                if len(carrying) > 0:
                    self.msgAboutDropLocation(state)
                    return "DropObject", {'object_id':self.agent_properties['is_carrying'][0]['obj_id'] } 
                for block in blocks:
                    if block['location'] == location:
                        self._sendGrabBlockMessage(state)
                        return "GrabObject", {'object_id':block['obj_id'] } 
                    
                id = list(self.collectBlocks.keys())
                id.sort()
                (x,y) = self.collectBlocks[id[0]]['location']
                if not state[self.agent_id]['location'] == (x-1,y):
                    self._phase=Phase.REMOVE_ALL_BLOCKS
                    return 'MoveSouth', {}   
                else:
                    self._phase=Phase.REPLACE_ALL_BLOCKS
                    return 'MoveEast', {}   
                
            if Phase.REPLACE_ALL_BLOCKS == self._phase:
                self.updateBlocks(state)
                (x,y) = state[self.agent_id]['location']
                
                blocks = self.detectBlocksAround(state)
                carrying = self.agent_properties['is_carrying']

                carrying = self.agent_properties['is_carrying']
                if len(carrying) > 0:
                    self.msgAboutDropLocation(state)
                    return "DropObject", {'object_id':self.agent_properties['is_carrying'][0]['obj_id'] } 
                
                for collectBlock in self.collectBlocks.values():
                    if collectBlock['location'] == (x,y) and not collectBlock['is_delivered_confirmed']:
                        for block in blocks:
                            if self.sameVizuals(block, collectBlock) and block['location'] == (x-1,y):
                                self._sendGrabBlockMessage(state)
                                return "GrabObject", {'object_id':block['obj_id'] } 
                    if collectBlock['location'] == (x,y) and collectBlock['is_delivered_confirmed']:
                        return "MoveNorth", {}
                self._phase=Phase.PLAN_TO_GOAL_BLOCK 
                
                    
    def checkAllCollectBlocksPresent(self):
        for collectBlock in self.collectBlocks.values():
            if collectBlock['is_delivered_confirmed'] == False:
                return False
        return True
    
    def getReachableLocations(self, state:State):
        (x, y )=  state[self.agent_id]['location']
        return [(x-1,  y), (x,  y), (x+1, y), (x,  y-1), (x,  y+1)]

    def updateCollect(self, state:State):
        for collectBlock in self.collectBlocks.values():
            blockFound = False
            # location = state[self.agent_id]['location']
            # if location == collectBlock['location']:
            if collectBlock['location'] in self.getReachableLocations(state):
                for block in self.detectBlocksAround(state):
                    if (block['location'] == collectBlock['location']) and self.sameVizuals(collectBlock, block):
                        self.collectBlocks[collectBlock['obj_id']]['is_delivered_confirmed'] = True
                        self.collectBlocks[collectBlock['obj_id']]['is_delivered_by_me'] = True
                        blockFound = True
                if not blockFound:
                    self.collectBlocks[collectBlock['obj_id']]['is_delivered_confirmed'] = False
                    self.collectBlocks[collectBlock['obj_id']]['is_delivered_by_me'] = False
                        
                        
    
    def _planPathToUnsearchedRoom(self):
        self._navigator.reset_full()
        # Randomly pick a closed door
        self._door = self.roomsToExplore[-1]
        self.roomsToExplore.remove(self._door)
        doorLoc = self._door['location']
        # Location in front of door is south from door
        doorLoc = doorLoc[0],doorLoc[1]+1
        # Send message of current action
        self._navigator.add_waypoints([doorLoc])  
        
    def getBlockToGrab(self):
        actions = super()._blockActions(MessageType.PICKING_UP)
        possibleCollectedBlocks = []
        ids = list(self.knownBlocks.keys())
        ids.sort(reverse=True)
        for _collectBlock in self.collectBlocks.values():
            if not _collectBlock['is_delivered_by_me'] or not _collectBlock['is_delivered_confirmed']:
                collectBlock = _collectBlock
                
                if collectBlock is None:
                    return None
                for id in ids:
                    block = self.knownBlocks[id]
                    if block['isGoalBlock'] and block['is_delivered'] == False and self.sameVizuals(collectBlock, block):
                        possibleRelevantActions = []
                        for action in actions:
                            if block['location'] == action[1] and super()._trustInAgent(agent_id= action[2]):
                                possibleRelevantActions.append(action)
                                break
                        if len(possibleRelevantActions) > 0:
                            possibleCollectedBlocks.append(block)
                        else:
                            self.blockToGrab = block
                            return block
                            
                if(len(possibleCollectedBlocks) > 0):
                    self.blockToGrab = possibleCollectedBlocks[-1]
                    return  possibleCollectedBlocks[-1]
                return None
                    
    def _possibleToPlanPathToGoalBlock(self):
        self._navigator.reset_full()
        collectBlock = self.getBlockToGrab()
        if collectBlock is None:
            return False
        self._navigator.add_waypoints([self.blockToGrab['location']])
        return True
        
              
    def _planPathToDropOff(self):
        self._navigator.reset_full()
        carriedBlock = self.agent_properties['is_carrying'][0]
        location = (0, 0)
        ids = list(self.collectBlocks.keys())
        ids.sort()
        
        for name in ids:            
            if (self.sameVizuals(self.collectBlocks[name], carriedBlock) and not self.collectBlocks[name]['is_delivered_confirmed']): 
                location = self.collectBlocks[name]["location"]
                break
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
        msg = random.choice([door for door in state.values()
            if 'class_inheritance' in door and 'Door' in door['class_inheritance'] 
            and door['room_name'] is not self._door['room_name']])['room_name'] if self.toLieOrNotToLieZetsTheKwestion() else self._door['room_name']
        super()._sendMessage("Searching through "  + str(msg), state[self.agent_id]['obj_id']) 
        
    def _sendMovingToDoorMessage(self, state:State, correctDoor):       
        msg = random.choice([door for door in state.values()
            if 'class_inheritance' in door and 'Door' in door['class_inheritance'] 
            and door['room_name'] is not correctDoor])['room_name'] if self.toLieOrNotToLieZetsTheKwestion() else correctDoor
        super()._sendMessage('Moving to ' + str(msg), state[self.agent_id]['obj_id'])
            
    def _sendDoorOpenMessage(self, state:State):
        door = random.choice([door for door in state.values()
                    if 'class_inheritance' in door and 'Door' in door['class_inheritance'] 
                    and door['room_name'] is not self._door['room_name']]) if self.toLieOrNotToLieZetsTheKwestion() else self._door
        super()._sendMessage('Opening door of ' + str(door['room_name']), state[self.agent_id]['obj_id'])
         
    def sendGoalBlockFoundMessage(self, state:State, block):
        toLie = self.toLieOrNotToLieZetsTheKwestion()
        lieOptions = [otherBlock for otherBlock in self.collectBlocks.values()
                    if not self.sameVizuals(block, otherBlock)]
        location = state[block['obj_id']]['location']
        if len(lieOptions) > 0:
            lie = random.choice(lieOptions) 
            
        else: 
            lie = state[block['obj_id']]
            location = random.choice([otherBlock for otherBlock in self.knownBlocks.values()])['location'] if toLie else location
        messageBlock = lie if toLie else state[block['obj_id']] 
        msg = "Found goal block " + str({"size": messageBlock["visualization"]['size'],
                                         "shape":  messageBlock["visualization"]['shape'],
                                         "colour":  messageBlock["visualization"]['colour']}) + " at location " + str(location)
        super()._sendMessage(msg, state[self.agent_id]['obj_id'])
    
    def _sendGrabBlockMessage(self, state:State):
        lie = self.toLieOrNotToLieZetsTheKwestion()
        block = self.blockToGrab
        location = self.blockToGrab['location']
        if lie and len(self.knownBlocks) > 1 and len([block for block_id in self.knownBlocks
            if not self.sameVizuals(self.knownBlocks[block_id], self.blockToGrab) and block['isGoalBlock']]) > 0:
            block = random.choice([block for block_id in self.knownBlocks
            if not self.sameVizuals(self.knownBlocks[block_id], self.blockToGrab) and block['isGoalBlock']])
            location = block['location']                
            
        elif lie:
            location = random.choice([otherBlock for otherBlock in self.knownBlocks.values()])['location']
        msg = "Picking up goal block " + str({"size":  block['visualization']['size'], 
                                                         "shape": block['visualization']['shape'],
                                                         "colour": block['visualization']['colour']}) + " at location " + str(location)   
        super()._sendMessage(msg, state[self.agent_id]['obj_id'])     
        
    def msgAboutDropLocation(self, state:State):
        carriedBlock = self.agent_properties['is_carrying'][0]
        lie = self.toLieOrNotToLieZetsTheKwestion()
        location = state[self.agent_id]['location']
        block = carriedBlock
        
        if lie: 
            if len(self.collectBlocks) > 0 and len([block for block in self.collectBlocks.values()
                    if  (block['visualization']['shape']  is not carriedBlock['visualization']['shape']) or
                        (block['visualization']['colour'] is not carriedBlock['visualization']['colour']) or
                        (block['visualization']['size']   is not carriedBlock['visualization']['size'])]) > 0:
                block = random.choice([block for block in self.collectBlocks.values()
                    if  (block['visualization']['shape']  is not carriedBlock['visualization']['shape']) or
                        (block['visualization']['colour'] is not carriedBlock['visualization']['colour']) or
                        (block['visualization']['size']   is not carriedBlock['visualization']['size'])])
        msg = "Dropped goal block " + str({"size":  block['visualization']['size'] 
                                                        , "shape": block['visualization']['shape']
                                                        , "colour": block['visualization']['colour']}) + " at drop location " + str(location)        
        super()._sendMessage(msg, state[self.agent_id]['obj_id'])      
                
    def updateBlock(self, block):
        obj_id = block['obj_id']
        if obj_id in self.knownBlocks.keys():
            self.knownBlocks[obj_id]['location'] = block['location']
            for collectBlock in self.collectBlocks.values():
                if self.sameVizuals(block, collectBlock):
                    if block['location'] == collectBlock['location']:
                        self.knownBlocks[obj_id]['is_delivered'] = True
                        self.knownBlocks[obj_id]['is_delivered_confirmed'] = True
                        self.collectBlocks[collectBlock['obj_id']]['is_delivered_confirmed'] = True
                        break
                    else:
                        self.knownBlocks[obj_id]['is_delivered'] = False
                        self.knownBlocks[obj_id]['is_delivered_confirmed'] = False

                                                    
            
    '''
    Update existing blocks
    '''
    def updateBlocks(self, state:State):
        self.updateCollect(state)
        for block in self.detectBlocksAround(state):
            self.addNewBlock(state, block)
            self.updateBlock(block)
                               
            
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
        #     super()._sendMessage("My next message is a lie", self.state[self.agent_id]['obj_id'])
        return lie
    
    def _roomExplorationWayPoints(self, state:State):
        self._navigator.reset_full()
        door = self._door
        room = self._getRoomSize(door['room_name'], state)
        waypoints = [(room[1][0]-1,room[1][1])]
        currentX = room[1][0]-1
        currentY = room[1][1]
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
    
    def _validateBlock(self, location, color: str, shape: int): 
        possible_blocks = []
        for block in self.knownBlocks.values():
            if (block['location'] == location):
                possible_blocks.append(block)
                if color == "?" and block['visualization']['shape'] == shape: 
                    return 1
                elif (block['visualization']['colour'] == color and 
                    block['visualization']['shape'] == shape):
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






class Strong(BaseLineAgent):

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
                        self.collectBlocks[key]['is_carried_by_me'] = False
                        self.collectBlocks[key]['is_delivered_by_me'] = False # dropActions = {'agent': None, 'number': None}
                self.roomsToExplore = [door for door in state.values()
                    if 'class_inheritance' in door and 'Door' in door['class_inheritance']] 
                         
                self._phase=Phase.PLAN_PATH_TO_UNSEARCHED_ROOM
            if Phase.PLAN_PATH_TO_UNSEARCHED_ROOM==self._phase:
                self.updateBlocks(state)
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
                self.updateBlocks(state)
                self._phase=Phase.PLAN_ROOM_EXPLORATION                
                if not self._door['is_open']:
                    for block in state.keys():
                        if "door" in block and state[block]["obj_id"] == self._door['obj_id'] and not state[block]["is_open"]:
                            self._sendDoorOpenMessage(state)
                            return "OpenDoorAction" , {'object_id':self._door['obj_id']}  
            
            elif Phase.PLAN_ROOM_EXPLORATION==self._phase:
                self.updateBlocks(state)
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
                self.updateBlocks(state)
                possible = self._possibleToPlanPathToGoalBlock()
                if possible == False:
                    self._phase=Phase.PLAN_PATH_TO_VERIFY_COLLECTION
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
                self.updateBlocks(state)
                blocks = self.detectBlocksAround(state)
                ids = []
                for block in blocks:
                    ids.append(block['obj_id'])
                if self.blockToGrab['obj_id'] not in ids:
                    #BLOCK NOT ON LAST KNOWN LOCATION
                    del self.knownBlocks[self.blockToGrab['obj_id']]
                    if len(self.roomsToExplore) > 0:
                        self._phase = Phase.PLAN_PATH_TO_UNSEARCHED_ROOM
                        continue
                    self._phase = Phase.PLAN_TO_GOAL_BLOCK
                    continue       
                
                self._sendGrabBlockMessage(state)
                self._phase=Phase.PLAN_TO_DROP_ZONE
                return "GrabObject", {'object_id':self.blockToGrab['obj_id'] } 
            
            if Phase.PLAN_TO_DROP_ZONE==self._phase:
                self.updateBlocks(state)
                if(len(self.agent_properties['is_carrying']) == 0):
                    #NO BLOCKS BRAPPED
                    self._phase = Phase.PLAN_PATH_TO_UNSEARCHED_ROOM
                    continue
                ids = self.collectBlocks.keys()
                # ids.sort()
                for block in self.agent_properties['is_carrying']:
                    
                    for id in self.collectBlocks.keys():
                        collectBlock = self.collectBlocks[id]
                        if self.sameVizuals(collectBlock, block) and not collectBlock['is_carried_by_me'] and not self.knownBlocks[block['obj_id']]['is_carried_by_me']:
                            self.collectBlocks[id]['is_carried_by_me'] = True
                            self.knownBlocks[block['obj_id']]['is_carried_by_me'] = True
                            break
                
                if self._possibleToPlanPathToGoalBlock():
                    self._phase=Phase.PLAN_TO_GOAL_BLOCK
                else:    
                    self._planPathToDropOff()
                    self._phase=Phase.FOLLOW_PATH_TO_DROP_ZONE
            
            if Phase.FOLLOW_PATH_TO_DROP_ZONE==self._phase:
                self.updateBlocks(state)
                self.updateCollect(state)
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action, {}   
                self._phase=Phase.PLAN_PATH_TO_VERIFY_COLLECTION
                self.processDropGoalBlockAtCollectPoint(state)
                self.msgAboutDropLocation(state)
                self.collectBlocks[self.getCollectBlockIdForCarriedBlock()]['is_carried_by_me'] = False # possibly -1 instead of 0
                
                if(len(self.agent_properties['is_carrying']) == 1):
                    self._phase=Phase.PLAN_PATH_TO_VERIFY_COLLECTION
                else:
                    self._phase=Phase.DOUBLE_DROPPOFF
                return "DropObject", {'object_id':self.getFirstCarriedBlock()['obj_id'] }    
            
            if Phase.DOUBLE_DROPPOFF==self._phase:
                self.updateBlocks(state)
                if(len(self.agent_properties['is_carrying']) == 0):
                    #NO BLOCKS BRAPPED
                    self._phase = Phase.PLAN_PATH_TO_UNSEARCHED_ROOM
                    continue
                ids = self.collectBlocks.keys()

                self._planPathToDropOff()
                self._phase=Phase.FOLLOW_PATH_TO_DROP_ZONE
                
            if Phase.PLAN_PATH_TO_VERIFY_COLLECTION==self._phase:
                self.updateBlocks(state)
                self.updateCollect(state)
                self._navigator.reset_full()
                locations = []
                for collectBlock in self.collectBlocks.values():                 
                    locations.append(collectBlock['location'])
                self._navigator.add_waypoints(locations)
                self._phase=Phase.FOLLOW_PATH_TO_VERIFY_COLLECTION
                
            if Phase.FOLLOW_PATH_TO_VERIFY_COLLECTION==self._phase:
                self.updateBlocks(state)
                self.updateCollect(state)
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action, {}   
                
                if self.checkAllCollectBlocksPresent():
                    self._phase=Phase.PLAN_PATH_TO_REMOVE_ALL_BLOCKS
                elif len (self.roomsToExplore) > 0:
                    self._phase=Phase.PLAN_PATH_TO_UNSEARCHED_ROOM
                else:
                    self._phase=Phase.PLAN_TO_GOAL_BLOCK
                    
            if Phase.PLAN_PATH_TO_REMOVE_ALL_BLOCKS==self._phase:
                self.updateBlocks(state)
                self.updateCollect(state)
                id = list(self.collectBlocks.keys())
                id.sort(reverse=True)
                (x,y) = self.collectBlocks[id[0]]['location']
                location = (x-1,y)
                
                self._navigator.reset_full()
                self._navigator.add_waypoints([location])
                
                self._phase = Phase.FOLLOW_PATH_TO_REMOVE_ALL_BLOCKS
                
            if Phase.FOLLOW_PATH_TO_REMOVE_ALL_BLOCKS==self._phase:
                self.updateBlocks(state)
                self.updateCollect(state)
                
                self._state_tracker.update(state)
                
                action = self._navigator.get_move_action(self._state_tracker)
                
                if action!=None:
                    return action, {}   
                self._phase = Phase.REMOVE_ALL_BLOCKS
            
            if Phase.REMOVE_ALL_BLOCKS == self._phase:
                self.updateBlocks(state)
                self.updateCollect(state)
                (x,y) = state[self.agent_id]['location']
                location = (x+1,y)
                blocks = self.detectBlocksAround(state)

                carrying = self.agent_properties['is_carrying']

                if len(carrying) > 0:
                    self.msgAboutDropLocation(state)
                    return "DropObject", {'object_id':self.agent_properties['is_carrying'][0]['obj_id'] } 
                for block in blocks:
                    if block['location'] == location:
                        self._sendGrabBlockMessage(state)
                        return "GrabObject", {'object_id':block['obj_id'] } 
                    
                id = list(self.collectBlocks.keys())
                id.sort()
                (x,y) = self.collectBlocks[id[0]]['location']
                if not state[self.agent_id]['location'] == (x-1,y):
                    self._phase = Phase.REMOVE_ALL_BLOCKS
                    return 'MoveSouth', {}   
                else:
                    self._phase = Phase.REPLACE_ALL_BLOCKS
                    return 'MoveEast', {}   
                
            if Phase.REPLACE_ALL_BLOCKS == self._phase:
                self.updateBlocks(state)
                self.updateCollect(state)
                (x,y) = state[self.agent_id]['location']
                
                blocks = self.detectBlocksAround(state)
                carrying = self.agent_properties['is_carrying']

                carrying = self.agent_properties['is_carrying']
                if len(carrying) > 0:
                    self.msgAboutDropLocation(state)
                    return "DropObject", {'object_id':self.agent_properties['is_carrying'][0]['obj_id'] } 
                
                for collectBlock in self.collectBlocks.values():
                    if collectBlock['location'] == (x,y) and not collectBlock['is_delivered_confirmed']:
                        for block in blocks:
                            if self.sameVizuals(block, collectBlock) and block['location'] == (x-1,y):
                                self._sendGrabBlockMessage(state)
                                return "GrabObject", {'object_id':block['obj_id'] } 
                    if collectBlock['location'] == (x,y) and collectBlock['is_delivered_confirmed']:
                        return "MoveNorth", {}
                self._phase = Phase.PLAN_TO_GOAL_BLOCK 
                
    def getFirstCarriedBlock(self):
        carrying = self.agent_properties['is_carrying']
        if len(carrying) == 2:
            return self.agent_properties['is_carrying'][0]
        elif len(carrying) == 1:
            return self.agent_properties['is_carrying'][0] 
        return None
        
        
    def getCollectBlockIdForCarriedBlock(self):
        carrying = self.getFirstCarriedBlock()
        ids = list(self.collectBlocks.keys())
        ids.sort()
        for id in ids:
            collectBlock = self.collectBlocks[id]
            if self.sameVizuals(collectBlock, carrying) and collectBlock['is_carried_by_me']:
                return collectBlock['obj_id']
        return None
                
                    
    def checkAllCollectBlocksPresent(self):
        for collectBlock in self.collectBlocks.values():
            if collectBlock['is_delivered_confirmed'] == False:
                return False
        return True
    
    def getReachableLocations(self, state:State):
        (x, y )=  state[self.agent_id]['location']
        return [(x-1,  y), (x,  y), (x+1, y), (x,  y-1), (x,  y+1)]

    def updateCollect(self, state:State):
        for collectBlock in self.collectBlocks.values():
            blockFound = False
            # location = state[self.agent_id]['location']
            # if location == collectBlock['location']:
            if collectBlock['location'] in self.getReachableLocations(state):
                for block in self.detectBlocksAround(state):
                    if (block['location'] == collectBlock['location']) and self.sameVizuals(collectBlock, block):
                        self.collectBlocks[collectBlock['obj_id']]['is_delivered_confirmed'] = True
                        self.collectBlocks[collectBlock['obj_id']]['is_delivered_by_me'] = True
                        blockFound = True
                if not blockFound:
                    self.collectBlocks[collectBlock['obj_id']]['is_delivered_confirmed'] = False
                    self.collectBlocks[collectBlock['obj_id']]['is_delivered_by_me'] = False
                        
                        
    
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
            if (not _collectBlock['is_delivered_by_me'] or not _collectBlock['is_delivered_confirmed'] )and not  _collectBlock['is_carried_by_me']:
                collectBlock = _collectBlock
                
                if collectBlock is None:
                    return None
                ids = list(self.knownBlocks.keys())
                ids.sort(reverse=True)
                for id in ids:
                    block = self.knownBlocks[id]
                    if block['isGoalBlock'] and block['is_delivered'] == False and self.sameVizuals(collectBlock, block) and not block['is_carried_by_me'] :
                        self.blockToGrab = block
                        return block
                return None
                    
    def _possibleToPlanPathToGoalBlock(self):
        if (len(self.agent_properties['is_carrying']) == 2):
            return False
        self._navigator.reset_full()
        collectBlock = self.getBlockToGrab()
        if collectBlock is None:
            return False
        self._navigator.add_waypoints([self.blockToGrab['location']])
        return True
        
              
    def _planPathToDropOff(self):
        self._navigator.reset_full()
        carriedBlock = self.getFirstCarriedBlock()
        location = (0, 0)
        ids = list(self.collectBlocks.keys())
        ids.sort()
        
        for name in ids:            
            if (self.sameVizuals(self.collectBlocks[name], carriedBlock) and not self.collectBlocks[name]['is_delivered_confirmed']): 
                location = self.collectBlocks[name]["location"]
                break
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
                    self.knownBlocks[obj_id]['is_carried_by_me'] = False                   
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
        super()._sendMessage("Searching through "  + str(msg), state[self.agent_id]['obj_id']) 
        
    def _sendMovingToDoorMessage(self, state:State, correctDoor):       
        msg = correctDoor
        super()._sendMessage('Moving to ' + str(msg), state[self.agent_id]['obj_id'])
            
    def _sendDoorOpenMessage(self, state:State):
        door = self._door
        super()._sendMessage('Opening door of ' + str(door['room_name']), state[self.agent_id]['obj_id'])
         
    def sendGoalBlockFoundMessage(self, state:State, block):
        location = state[block['obj_id']]['location']
        messageBlock = state[block['obj_id']] 
        msg = "Found goal block " + str({"size": messageBlock["visualization"]['size'],
                                         "shape":  messageBlock["visualization"]['shape'],
                                         "colour":  messageBlock["visualization"]['colour']}) + " at location " + str(location)
        super()._sendMessage(msg, state[self.agent_id]['obj_id'])
    
    def _sendGrabBlockMessage(self, state:State):
        block = self.blockToGrab
        location = self.blockToGrab['location']
        msg = "Picking up goal block " + str({"size":  block['visualization']['size'], 
                                                         "shape": block['visualization']['shape'],
                                                         "colour": block['visualization']['colour']}) + " at location " + str(location)   
        super()._sendMessage(msg, state[self.agent_id]['obj_id'])     
        
    def msgAboutDropLocation(self, state:State):
        carriedBlock = self.getFirstCarriedBlock()
        location = state[self.agent_id]['location']
        block = carriedBlock
        msg = "Dropped goal block " + str({"size":  block['visualization']['size'] 
                                                        , "shape": block['visualization']['shape']
                                                        , "colour": block['visualization']['colour']}) + " at drop location " + str(location)        
        super()._sendMessage(msg, state[self.agent_id]['obj_id'])      
                
    def updateBlock(self, block):
        obj_id = block['obj_id']
        if obj_id in self.knownBlocks.keys():
            self.knownBlocks[obj_id]['location'] = block['location']
            for collectBlock in self.collectBlocks.values():
                if self.sameVizuals(block, collectBlock):
                    if block['location'] == collectBlock['location']:
                        self.knownBlocks[obj_id]['is_delivered'] = True
                        self.knownBlocks[obj_id]['is_delivered_confirmed'] = True
                        self.collectBlocks[collectBlock['obj_id']]['is_delivered_confirmed'] = True
                        break
                    else:
                        self.knownBlocks[obj_id]['is_delivered'] = False
                        self.knownBlocks[obj_id]['is_delivered_confirmed'] = False

                                                    
            
    '''
    Update existing blocks
    '''
    def updateBlocks(self, state:State):
        for block in self.detectBlocksAround(state):
            self.addNewBlock(state, block)
            self.updateBlock(block)
                               
            
    def checkGoalBlockPresent(self, state:State):
        for block in state.keys():
            if "Block_in" in block and state[block]["location"] == state[self.agent_id]['location']:
                return True
        return False
    
    def processDropGoalBlockAtCollectPoint(self, state:State):
        carriedBlock = self.getFirstCarriedBlock()
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
    
    def _validateBlock(self, location, color: str, shape: int): 
        possible_blocks = []
        for block in self.knownBlocks.values():
            if (block['location'] == location):
                possible_blocks.append(block)
                if color == "?" and block['visualization']['shape'] == shape: 
                    return 1
                elif (block['visualization']['colour'] == color and 
                    block['visualization']['shape'] == shape):
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
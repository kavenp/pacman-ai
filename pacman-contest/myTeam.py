# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random
import time
import util
from game import Directions
import game

#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='DummyAgent', second='DummyAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''
        self.s_table = SarsaTable()

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)
        '''
        You should change this in your own agent.
        '''
        curr_pos = gameState.getAgentPosition(self.index)

        for episode in range(10):
            self.s_table.addNewState(curr_pos, actions)
            chosen_action = self.s_table.chooseAction(curr_pos)
            while True:
                start = time.time()
                next_state = gameState.generateSuccessor(self.index, chosen_action)
                next_pos = next_state.getAgentState(self.index).getPosition()
                next_action = self.s_table.chooseAction(next_pos)
                reward = self.evaluate(next_state, next_action)
                self.s_table.learn(self.getPacmanPosition(), chosen_action, reward, next_pos, next_action)
                curr_pos = next_pos
                chosen_action = next_action

                if (time.time() - start) >= 0.1:
                    break

        return chosen_action

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1}


class SarsaTable():
    def __init__(self, learning_rate=0.01, reward_decay_rate=0.9, e_greedy=0.9):
        self.lr = learning_rate
        self.dr = reward_decay_rate
        self.epsilon = e_greedy
        self.sarsa_table = {}

    def addNewState(self, agent_pos, actions):
        if agent_pos not in self.sarsa_table:
            self.sarsa_table[agent_pos] = {}
            for i in actions:
                self.sarsa_table[agent_pos][i] = 0

    def learn(self, curr_pos, curr_action, reward, next_pos, next_action):
        self.addNewState(next_pos,)
        q_assumed = self.sarsa_table[curr_pos][curr_action]
        q_real = reward + self.dr * self.sarsa_table[next_pos][next_action]
        self.sarsa_table[curr_pos][curr_action] += self.lr * (q_real - q_assumed)

    def chooseAction(self, curr_pos):
        action = max(self.sarsa_table[curr_pos])
        return action

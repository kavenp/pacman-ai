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
from math import *
import random, time, util, sys, math
from game import Directions
import game
import sys
import copy
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first='OffensiveAgent', second='OffensiveAgent'):
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


class MonteCarloAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.startState = gameState
        # self.tree = TreeNode(parent_node=None, action=None, gameState=copy.deepcopy(gameState))
        # self.tree = TreeNode(None, None, copy.deepcopy(gameState), self.index)
        # self.currentNode = self.tree

    def chooseAction(self, gameState):
        tree = TreeNode(parent_node=None, action=None, gameState=copy.deepcopy(gameState), index=self.index)

        self.computationResource = 50

        while self.computationResource:
            currentNode = self.treePolicy(tree, 1)
            if currentNode is not None:
                reward = self.defaultPolicy(currentNode.gameState)
                currentNode.backup(reward)
                self.computationResource -= 1

        return tree.getMaxUCTChild().action

    def treePolicy(self, tree, para):
        while tree.gameState is not None:
            root = tree
            if not tree.isFullyExpand():
                childAction = random.choice(tree.actionsNeedsTry)
                return tree.expand(childAction)
            else:
                maxUCB = - sys.maxint - 1
                for childState in tree.childrenStates:
                    ucb = self.normalizedScoreEvaluation(tree.state, root.state) + (
                            para * math.sqrt(2 * math.log1p(tree.timesVisited) / childState.timesVisited))
                    if ucb > maxUCB:
                        maxUCB = ucb
                        maxState = childState
                return maxState
        return tree

    def defaultPolicy(self, gameState):
        rollout = 0
        while gameState is not None and rollout < 5:
            action = random.choice(gameState.getLegalActions(self.index))
            successor = gameState.generateSuccessor(self.index, action)
            if successor is not None:
                gameState = successor
            else:
                self.computationalBudgetLeft = False
            rollout = rollout + 1
        return self.scoreEvaluation(gameState)

    def manhattanDis(self, positionA, positionB):
        return abs(positionA[0] - positionB[0]) + abs(positionA[1] - positionB[1])

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def scoreEvaluation(self, state):
        return state.getScore() + [0, -1000.0][state.isLose()] + [0, 1000.0][state.isWin()]

    def normalizedScoreEvaluation(self, rootState, currentState):
        rootEval = self.selfscoreEvaluation(rootState);
        currentEval = self.scoreEvaluation(currentState);
        return (currentEval - rootEval) / 1000.0;


class OffensiveAgent(MonteCarloAgent):

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


class TreeNode:

    def __init__(self, parent_node, action, gameState, index):
        print(action)
        self.gameState = gameState
        self.index = index
        self.action = action
        self.parentNode = parent_node
        self.actionsNeedsTry = self.gameState.getLegalActions(self.index)
        self.childNodes = []
        self.timeVisited = 0
        self.reward = 0.0

    def addChild(self, child):
        self.childNodes.append(child)
        self.actionsNeedsTry.remove(child.action)

    def UCTCalculate(self, child):
        if child.timeVisited <= 0:
            return sys.maxint
        else:
            score = self.reward + sqrt(2 * log(self.timeVisited) / child.timeVisited)
            return score

    def getMaxUCTChild(self):
        max = -sys.maxint - 1
        maxChild = None
        for child in self.childNodes:
            score = self.UCTCalculate(child)
            if score > max:
                maxChild = child
        return maxChild

    def expand(self, action):
        child = TreeNode(parent_node=self,
                         action=action,
                         gameState=self.gameState.generateSuccessor(self.index, action),
                         index=self.index)
        self.childNodes.append(child)
        self.actionsNeedsTry.remove(action)
        return child

    def isLeafNode(self):
        if len(self.childNodes) > 0:
            return False
        else:
            return True

    def isFullyExpand(self):
        if len(self.actionsNeedsTry) > 0:
            return False
        else:
            return True

    def Update(self, score):
        self.timeVisited += 1
        self.reward += score

    def backup(self, reward):
        it = self
        while it is not None:
            it.timeVisited += 1
            it.reward += reward
            it = it.parentNode

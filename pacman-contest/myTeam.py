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
from util import nearestPoint, Stack


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first='OffensiveAgent', second='DeffensiveAgent'):
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


class DefaultAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        return random.choice(gameState.getLegalActions(self.index))


class MonteCarloAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.startState = gameState
        self.MAX_DEPTH = 4
        self.MAX_SIMULATION_DEPTH = 1
        self.DISCOUNT_RATE = 0.2

        if gameState.isOnRedTeam(self.index):
            self.middle = (gameState.data.layout.width - 2) / 2
        else:
            self.middle = ((gameState.data.layout.width - 2) / 2) + 1
        self.boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(self.middle, i):
                self.boundary.append((self.middle, i))

    def chooseAction(self, gameState):
        if gameState.getAgentPosition(self.index)[0] > self.middle - 2:
            return self.selectActionBaseOnTree(gameState)
        else:
            return self.selectActionBaseOnDis(gameState)

    def selectActionBaseOnDis(self, gameState):
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluateDis(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def selectActionBaseOnTree(self, gameState):
        self.explore_depth = 0
        tree = TreeNode(parent_node=None, action=None, gameState=copy.deepcopy(gameState), index=self.index, depth=0)
        all_leaf_nodes = self.treePolicy(tree)
        for node in all_leaf_nodes:
            self.doSimulation(node)

        for child in tree.childNodes:
            print child.action + ":" + str(child.reward) + " ",
        print " [" + tree.getBestRewardChild().action + "]"
        return tree.getBestRewardChild().action

    def treePolicy(self, tree):
        stack = Stack()
        stack.push(tree)
        leaf_nodes = []
        while not stack.isEmpty():
            current = stack.pop()
            if current.depth < self.MAX_DEPTH - 1:
                while not current.isFullyExpand():
                    childAction = random.choice(current.actionsNeedsTry)
                    child = current.expand(childAction)
                    child.reward = self.evaluate(child.gameState)
                    stack.push(child)
            else:
                leaf_nodes.append(current)
        return leaf_nodes

    def doSimulation(self, leaf_node):
        discount = 1
        gameState = leaf_node.gameState
        leaf_node.reward = self.evaluate(leaf_node.gameState)
        for i in range(self.MAX_SIMULATION_DEPTH):
            discount = discount * self.DISCOUNT_RATE
            actions = gameState.getLegalActions(self.index)
            gameState = gameState.generateSuccessor(self.index, random.choice(actions))
            leaf_node.reward = (1 - discount) * leaf_node.reward + discount * self.evaluate(gameState)
        leaf_node.backup(self.DISCOUNT_RATE)

    def evaluate(self, gameState):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState)
        weights = self.getWeights(gameState, features)
        return features * weights

    def getFeatures(self, gameState):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, features):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

    def evaluateDis(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getDisFeatures(gameState, action)
        weights = self.getDisWeights(gameState, action)
        return features * weights

    def getDisFeatures(self, gameState, action):
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

    def getDisWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1}

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


class OffensiveAgent(MonteCarloAgent):

    def getFeatures(self, gameState):
        features = util.Counter()
        myPosition = gameState.getAgentState(self.index).getPosition()
        foodList = self.getFood(gameState).asList()

        # [1]carrying food number
        features['carrying'] = gameState.getAgentState(self.index).numCarrying

        # [2]left food number
        features['foodLeft'] = len(foodList)

        # [3]Compute distance to the nearest food
        if len(foodList) > 0:
            minDistance = min([self.getMazeDistance(myPosition, food) for food in foodList])
            features['distanceToFood'] = minDistance

        # [4]Compute distance to closest ghost
        opponentsState = []
        for i in self.getOpponents(gameState):
            opponentsState.append(gameState.getAgentState(i))
        visible = filter(lambda x: not x.isPacman and x.getPosition() != None, opponentsState)
        if len(visible) > 0:
            positions = [agent.getPosition() for agent in visible]
            closest = min(positions, key=lambda x: self.getMazeDistance(myPosition, x))
            closestDist = self.getMazeDistance(myPosition, closest)
            if closestDist <= 5:
                # print(CurrentPosition,closest,closestDist)
                features['GhostDistance'] = closestDist

        # [5]Dead Corner
        actions = gameState.getLegalActions(self.index)
        if len(actions) <= 2:
            features['corner'] = 1
        else:
            features['corner'] = 0

        # [6]Compute distance to the nearest capsule
        capsuleList = self.getCapsules(gameState)
        if len(capsuleList) > 0:
            minCapsuleDistance = 99999
            for c in capsuleList:
                distance = self.getMazeDistance(myPosition, c)
                if distance < minCapsuleDistance:
                    minCapsuleDistance = distance
            features['distanceToCapsule'] = minCapsuleDistance
        else:
            features['distanceToCapsule'] = 0

        # [7]Compute the distance to the nearest boundary
        boundaryMin = sys.maxint
        for i in range(len(self.boundary)):
            disBoundary = self.getMazeDistance(myPosition, self.boundary[i])
            if (disBoundary < boundaryMin):
                boundaryMin = disBoundary
        features['returned'] = boundaryMin

        # [8] state score
        features['successorScore'] = gameState.getScore()

        # [9]
        features['distanceToEnemiesPacMan'] = 0

        return features

    def getWeights(self, gameState, features):
        # If opponent is scared, the agent should not care about GhostDistance
        successor = gameState
        # numOfFood = len(self.getFood(successor).asList())
        numOfCarrying = successor.getAgentState(self.index).numCarrying
        # CurrentPosition = successor.getAgentState(self.index).getPosition()
        # myself = successor.getAgentState.(self.index).isPacman
        opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        visible = filter(lambda x: not x.isPacman and x.getPosition() != None, opponents)
        if len(visible) > 0:
            for agent in visible:
                if agent.scaredTimer > 0:
                    if agent.scaredTimer > 12:
                        return {'successorScore': 110, 'distanceToFood': -10, 'distanceToEnemiesPacMan': 0,
                                'GhostDistance': -1, 'distanceToCapsule': 0, 'carrying': 350,
                                'returned': 10 - 3 * numOfCarrying, 'corner': 0, 'foodLeft': 0}

                    elif 6 < agent.scaredTimer < 12:
                        return {'successorScore': 110 + 5 * numOfCarrying, 'distanceToFood': -5,
                                'distanceToEnemiesPacMan': 0,
                                'GhostDistance': -15, 'distanceToCapsule': -10, 'carrying': 100,
                                'returned': -5 - 4 * numOfCarrying, 'corner': 0, 'foodLeft': 0
                                }

                # Visible and not scared
                else:
                    return {'successorScore': 110, 'distanceToFood': -10, 'distanceToEnemiesPacMan': 0,
                            'GhostDistance': 20, 'distanceToCapsule': -15, 'carrying': 0, 'returned': -15, 'corner': 0,
                            'foodLeft': 0
                            }

        # If I am not PacMan the enemy is a pacMan, I can try to eliminate him
        # Attacker only try to defence if it is close to it (less than 4 steps)
        enemiesPacMan = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        Range = filter(lambda x: x.isPacman and x.getPosition() != None, enemiesPacMan)
        if len(Range) > 0 and not gameState.getAgentState(self.index).isPacman:
            return {'successorScore': 0, 'distanceToFood': -1, 'distanceToEnemiesPacMan': -8,
                    'distanceToCapsule': 0, 'GhostDistance': 0,
                    'returned': 0, 'carrying': 10, 'corner': 0,
                    'foodLeft': 0}

        # Did not see anything
        return {'successorScore': 1000 + numOfCarrying * 3.5, 'distanceToFood': -7, 'GhostDistance': 0,
                'distanceToEnemiesPacMan': 0,
                'distanceToCapsule': -5, 'carrying': 350, 'returned': 5 - numOfCarrying * 3, 'corner': 0,
                'foodLeft': 0}


class DeffensiveAgent(MonteCarloAgent):
    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

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

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


class TreeNode:

    def __init__(self, parent_node, action, gameState, index, depth):
        self.gameState = gameState
        self.index = index
        self.action = action
        self.parentNode = parent_node
        self.actionsNeedsTry = self.gameState.getLegalActions(self.index)
        self.actionsNeedsTry.remove('Stop')
        self.choiceNum = len(self.actionsNeedsTry)
        self.childNodes = []
        self.timeVisited = 0
        self.reward = 0.0
        self.depth = depth

    def getBestRewardChild(self):
        max = -sys.maxint - 1
        bestChild = None
        for child in self.childNodes:
            if child.reward > max:
                max = child.reward
                bestChild = child
        return bestChild

    def expand(self, action):
        child = TreeNode(parent_node=self,
                         action=action,
                         gameState=self.gameState.generateSuccessor(self.index, action),
                         index=self.index,
                         depth=self.depth + 1)
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

    def backup(self, discount):
        if self.parentNode is not None:
            self.parentNode.reward = (1 - discount) * self.parentNode.reward + self.reward * discount
            self.parentNode.backup(discount)

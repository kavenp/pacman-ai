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


class MonteCarloAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        # parameters
        self.MAX_TREE_DEPTH = 6
        self.MAX_SIMULATION_DEPTH = 1
        self.DISCOUNT_RATE = 0.3

        # get the boundary tiles
        self.boundary = self.getBoundry(gameState)

    def chooseAction(self, gameState):
        pass

    def getBoundry(self, gameState):
        if gameState.isOnRedTeam(self.index):
            middle = (gameState.data.layout.width - 2) / 2
        else:
            middle = ((gameState.data.layout.width - 2) / 2) + 1

        boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(middle, i):
                boundary.append((middle, i))
        return boundary

    '''
    =================================[go to food]===========================================
    '''

    def selectActionBaseOnDisToFood(self, gameState):
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluateDis(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())
        if foodLeft <= 2:
            bestDist = sys.maxint
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def evaluateDis(self, gameState, action):
        features = self.getDisFeatures(gameState, action)
        weights = self.getDisWeights(gameState, action)
        return features * weights

    def getDisFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['gameStateScore'] = -len(foodList)
        # Compute distance to the nearest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        return features

    def getDisWeights(self, gameState, action):
        return {'gameStateScore': 100, 'distanceToFood': -1}

    '''
    ==========================================================================================
    '''

    '''
    ===============================[Monte Carlo Tree Search]==================================
    '''

    def selectActionBaseOnTree(self, gameState):
        # visibleGoasts = self.getVisibleGoasts(gameState)
        # if len(visibleGoasts) > 0:
        #     disToGoast = self.computeMinDisToGoast(gameState)
        #     if disToGoast == 1 and gameState.getAgentState(self.index).scaredTimer >= 6:
        #         pass

        start = time.time()
        self.explore_depth = 0
        tree = TreeNode(parent_node=None, action=None, gameState=copy.deepcopy(gameState), index=self.index, depth=0)
        all_leaf_nodes = self.treePolicy(tree)
        for node in all_leaf_nodes:
            self.doSimulation(node)

        for child in tree.childNodes:
            print child.action + ":" + str(child.reward) + " ",
        print " [" + tree.getBestRewardChild().action + "]",
        print ' time[%d: %.4f]' % (self.index, time.time() - start)
        return tree.getBestRewardChild().action

    def treePolicy(self, tree):
        stack = Stack()
        stack.push(tree)
        leaf_nodes = []
        while not stack.isEmpty():
            current = stack.pop()
            if current.depth < self.MAX_TREE_DEPTH - 1:
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

    '''
    ==========================================================================================
    '''

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getEnemieGoasts(self, gameState):
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        goast = filter(lambda x: not x.isPacman, opponents)
        return goast

    def getEnemiePacman(self, gameState):
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        pacman = filter(lambda x: x.isPacman, opponents)
        return pacman

    def getVisibleGoasts(self, gameState):
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        visible = filter(lambda x: not x.isPacman and x.getPosition() != None, opponents)
        return visible

    def getNearestGoast(self, gameState):
        pass

    def getVisiblePacman(self, gameState):
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        visible = filter(lambda x: x.isPacman and x.getPosition() != None, opponents)
        return visible

    def getEenemies(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        return enemies

    def getVisibleEnemies(self, gameState):
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        visible = filter(lambda x: x.getPosition() != None, opponents)
        return visible

    def evaluate(self, gameState):
        features = self.getFeatures(gameState)
        weights = self.getWeights(gameState, features)
        return features * weights

    def getFeatures(self, gameState):
        pass

    def getWeights(self, gameState, features):
        pass

    def getAgentPosition(self, gameState):
        return gameState.getAgentState(self.index).getPosition()

    def getMinDisToBoundary(self, gameState):
        myPosition = self.getAgentPosition(gameState)
        boundaryMin = sys.maxint
        for i in range(len(self.boundary)):
            disBoundary = self.getMazeDistance(myPosition, self.boundary[i])
            if (disBoundary < boundaryMin):
                boundaryMin = disBoundary
        return boundaryMin

    def getEnemieGoastScareTime(self, gameState):
        goasts = self.getEnemieGoasts(gameState)
        scare_time = 0
        for goast in goasts:
            scare_time = goast.scaredTimer
            break
        return scare_time

    def computeMinDisToGoast(self, gameState):
        myPosition = gameState.getAgentState(self.index)
        opponentsState = []
        for i in self.getOpponents(gameState):
            opponentsState.append(gameState.getAgentState(i))
        visible = filter(lambda x: not x.isPacman and x.getPosition() != None, opponentsState)
        if len(visible) > 0:
            positions = [agent.getPosition() for agent in visible]
            closest = min(positions, key=lambda x: self.getMazeDistance(myPosition, x))
            closestDist = self.getMazeDistance(myPosition, closest)
            return closestDist


'''
=======================
=== Offensive Agent ===
=======================
'''


class OffensiveAgent(MonteCarloAgent):

    def chooseAction(self, gameState):
        visibleGoasts = self.getVisibleGoasts(gameState)
        if gameState.getAgentState(self.index).isPacman:
            if len(visibleGoasts) <= 0 or self.getEnemieGoastScareTime(gameState) >= 15:
                return self.selectActionBaseOnDisToFood(gameState)
            else:
                return self.selectActionBaseOnTree(gameState)
        else:
            return self.selectActionBaseOnDisToFood(gameState)

    def getFeatures(self, gameState):
        features = util.Counter()
        myPosition = gameState.getAgentState(self.index).getPosition()
        foodList = self.getFood(gameState).asList()
        '''
        [1]--->carrying food
        '''
        features['carrying'] = gameState.getAgentState(self.index).numCarrying

        '''
        [2]--->food left
        '''
        features['foodLeft'] = len(foodList)

        '''
        [3]--->Compute distance to the nearest food
        '''
        if len(foodList) > 0:
            minDistance = min([self.getMazeDistance(myPosition, food) for food in foodList])
            features['distanceToFood'] = minDistance

        '''
        [4]--->Compute distance to closest ghost
        '''
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

        '''
        [5]--->Dead Corner
        '''
        actions = gameState.getLegalActions(self.index)
        if len(actions) <= 2:
            features['corner'] = 1
        else:
            features['corner'] = 0

        '''
        [6]--->Compute distance to the nearest capsule
        '''
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

        '''
        [7]--->(return)Compute the distance to the nearest boundary
        '''
        features['returned'] = self.getMinDisToBoundary(gameState)

        '''
        [8]--->state score
        '''
        features['gameStateScore'] = gameState.getScore()

        '''
        [9]--->distance to enemie pacman
        '''
        features['distanceToEnemiesPacMan'] = 0

        return features

    def getWeights(self, gameState, features):
        '''
        [1]--->carrying food
        [2]--->food left
        [3]--->Compute distance to the nearest food
        [4]--->Compute distance to closest ghost
        [5]--->Dead Corner
        [6]--->Compute distance to the nearest capsule
        [7]--->Compute the distance to the nearest boundary
        [8]--->state score
        [9]--->distance to enemie pacman
        '''
        numOfCarrying = gameState.getAgentState(self.index).numCarrying
        visibleGoasts = self.getVisibleGoasts(gameState)
        if len(visibleGoasts) > 0:
            for goast in visibleGoasts:
                if goast.scaredTimer > 0:
                    if goast.scaredTimer >= 15:
                        return {'gameStateScore': 110,
                                'distanceToFood': -10,
                                'distanceToEnemiesPacMan': 0,
                                'GhostDistance': -5,
                                'distanceToCapsule': 0,
                                'carrying': 350,
                                'returned': 10 - 3 * numOfCarrying,
                                'corner': -2,
                                'foodLeft': 0}

                    elif 6 < goast.scaredTimer < 15:
                        return {'gameStateScore': 110 + 5 * numOfCarrying,
                                'distanceToFood': -5,
                                'distanceToEnemiesPacMan': 50,
                                'GhostDistance': -15,
                                'distanceToCapsule': -1000,
                                'carrying': 100,
                                'returned': -5 - 4 * numOfCarrying,
                                'corner': -2,
                                'foodLeft': 0
                                }
                else:
                    return {'gameStateScore': 30,
                            'distanceToFood': -3,
                            'distanceToEnemiesPacMan': 0,
                            'GhostDistance': 60,
                            'distanceToCapsule': -15,
                            'carrying': 0,
                            'returned': -30,
                            'corner': -60,
                            'foodLeft': 0
                            }

        # visiblePacman = self.getVisiblePacman(gameState)
        # if len(visiblePacman) > 0 and not gameState.getAgentState(self.index).isPacman:
        #     return {'gameStateScore': 0, 'distanceToFood': -1, 'distanceToEnemiesPacMan': 0,  # -8,
        #             'distanceToCapsule': 0, 'GhostDistance': 0,
        #             'returned': 0, 'carrying': 10, 'corner': -2,
        #             'foodLeft': 0}

        return {'gameStateScore': 1000 + numOfCarrying * 3.5,
                'distanceToFood': -7,
                'GhostDistance': 0,
                'distanceToEnemiesPacMan': 0,
                'distanceToCapsule': -5,
                'carrying': 350,
                'returned': 5 - numOfCarrying * 3,
                'corner': -2,
                'foodLeft': 0}


'''
========================
=== Deffensive Agent ===
========================
'''


class DeffensiveAgent(MonteCarloAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        # parameters
        self.MAX_TREE_DEPTH = 6
        self.MAX_SIMULATION_DEPTH = 1
        self.DISCOUNT_RATE = 0.3

        # get the boundary tiles
        self.boundary = self.getBoundry(gameState)

        self.DenfendList = {}
        self.target = None
        self.lastObservedFood = None
        # Update probabilities to each patrol point.
        self.DefenceProbability(gameState)

    def DefenceProbability(self, gameState):
        """
        This method calculates the minimum distance from our patrol
        points to our pacdots. The inverse of this distance will
        be used as the probability to select the patrol point as
        target.
        """
        total = 0

        for position in self.boundary:
            food = self.getFoodYouAreDefending(gameState).asList()
            closestFoodDistance = min(self.getMazeDistance(position, f) for f in food)
            if closestFoodDistance == 0:
                closestFoodDistance = 1
            self.DenfendList[position] = 1.0 / float(closestFoodDistance)
            total += self.DenfendList[position]

        # Normalize.
        if total == 0:
            total = 1
        for x in self.DenfendList.keys():
            self.DenfendList[x] = float(self.DenfendList[x]) / float(total)

    def selectPatrolTarget(self):
        """
        Select some patrol point to use as target.
        """

        maxProb = max(self.DenfendList[x] for x in self.DenfendList.keys())
        bestTarget = filter(lambda x: self.DenfendList[x] == maxProb, self.DenfendList.keys())
        return random.choice(bestTarget)

    def chooseAction(self, gameState):

        # start = time.time()

        DefendingList = self.getFoodYouAreDefending(gameState).asList()
        if self.lastObservedFood and len(self.lastObservedFood) != len(DefendingList):
            self.DefenceProbability(gameState)
        myPos = gameState.getAgentPosition(self.index)
        if myPos == self.target:
            self.target = None

        # Visible enemy , keep chasing.
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        inRange = filter(lambda x: x.isPacman and x.getPosition() != None, enemies)
        if len(inRange) > 0:
            eneDis, enemyPac = min([(self.getMazeDistance(myPos, x.getPosition()), x) for x in inRange])
            self.target = enemyPac.getPosition()
            # for x in inRange:
            # if self.agent.getMazeDistance(myPos,x.getPosition())==closestGhost:
            # self.target=x.getPosition()
            # print(self.target)

        elif self.lastObservedFood != None:
            eaten = set(self.lastObservedFood) - set(self.getFoodYouAreDefending(gameState).asList())
            if len(eaten) > 0:
                closestFood, self.target = min([(self.getMazeDistance(myPos, f), f) for f in eaten])

        self.lastObservedFood = self.getFoodYouAreDefending(gameState).asList()

        # We have only a few dots.
        if self.target == None and len(self.getFoodYouAreDefending(gameState).asList()) <= 4:
            food = self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(
                gameState)
            self.target = random.choice(food)

        # Random patrolling
        elif self.target == None:
            self.target = self.selectPatrolTarget()

        actions = gameState.getLegalActions(self.index)
        feasible = []
        fvalues = []
        for a in actions:
            new_state = gameState.generateSuccessor(self.index, a)
            if not a == Directions.STOP and not new_state.getAgentState(self.index).isPacman:
                newPosition = new_state.getAgentPosition(self.index)
                feasible.append(a)
                fvalues.append(self.getMazeDistance(newPosition, self.target))

        # Randomly chooses between ties.
        best = min(fvalues)
        ties = filter(lambda x: x[0] == best, zip(fvalues, feasible))

        # print 'eval time for defender agent %d: %.4f' % (self.index, time.time() - start)
        return random.choice(ties)[1]


'''
=================
=== Tree Node ===
=================
'''


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

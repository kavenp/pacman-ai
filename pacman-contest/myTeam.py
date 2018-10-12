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

import game, random, time, util, sys
from captureAgents import CaptureAgent
from game import Directions

#################
# Team creation #
#################



def createTeam(firstIndex, secondIndex, isRed,
               first = 'GameAgent', second = 'GameAgent'):
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

class GameAgent(CaptureAgent):
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
    
    
    # Initialise map size
    self.mapW = gameState.data.layout.width
    self.mapH = gameState.data.layout.height
    
    # Initialise team and enemy
    self.team = self.getTeam(gameState)
    self.enemies = self.getOpponents(gameState)

    self.seen = []

    # Initialise strategy for team as Attack
    self.strats = {}
    for i in self.team:
        self.strats[i] = 'Attack'

    # Flags for agent choice status
    # More detailed than strategy

    self.closestEnemy = None
    self.agentPos = gameState.getAgentPosition(self.index)
    #self.closestFood = self.getClosestFood(self.foodList, self.agentPos)[0]

    # List of possible positions agents can be in
    self.legalPositions = gameState.getWalls().asList(False)

    # Initialise beliefs
    self.beliefs = {}
    for enemyAgent in self.enemies:
      self.beliefs[enemyAgent] = util.Counter()
      # We know where enemies start, so we set the initial belief here to 1
      self.beliefs[enemyAgent][gameState.getInitialAgentPosition(enemyAgent)] = 1.0

  def initBeliefs(self, enemy):
    """
    Initialise beliefs to uniform distribution
    """
    self.beliefs[enemy] = util.Counter()
    for pos in self.legalPositions:
      self.beliefs[enemy][pos] = 1.0
    # Normalise to uniform distribution
    self.beliefs[enemy].normalize()

  def elapseTime(self, enemy, gameState):
    """
    Include information on how the enemy can move over time
    """
    newBeliefs = util.Counter()

    for prevPos in self.legalPositions:
      newDist = util.Counter()
      stepSize = [-1, 0, 1]
      # Calculate set of possible moves for enemy
      movePos = []
      for i in stepSize:
        for j in stepSize:
          movePos += [(prevPos[0] + i, prevPos[1] + j)]
      # Check if moves are legal
      for pos in movePos:
        if pos in self.legalPositions:
          newDist[pos] = 1.0
      newDist.normalize()
      # Calculate new distribution
      for nextPos, p in newDist.items():
        # Update position probabilities
        newBeliefs[nextPos] += p * self.beliefs[enemy][prevPos]
    # Update beliefs to new
    newBeliefs.normalize()
    self.beliefs[enemy] = newBeliefs

  def observe(self, enemy, gameState):
    """
    Use Hidden Markov Model to guess enemy position
    """
    obsDist = gameState.getAgentDistances()[enemy]
    agentPos = gameState.getAgentPosition(self.index)
    # Create new dictionary to hold the new beliefs for the current enemy.
    newBeliefs = util.Counter()
    # Calculate new belief for each position
    for pos in self.legalPositions:
      # Calculating manhattan distance to the position
      mdist = util.manhattanDistance(agentPos, pos)
      # Emission probability
      emission = gameState.getDistanceProb(mdist, obsDist)
      # If within range 5 it should be accurate distance not fuzzy
      if mdist < 6:
        newBeliefs[pos] = 0.0
      else:
        # Update belief
        newBeliefs[pos] = self.beliefs[enemy][pos] * emission

    if newBeliefs.totalCount() == 0:
      self.initBeliefs(enemy)
    else:
      # Normalize and set the new belief.
      newBeliefs.normalize()
      self.beliefs[enemy] = newBeliefs

  def chooseAction(self, gameState):
    agentPos = gameState.getAgentPosition(self.index)
    # Inferred gamestate copy where we will place enemies most likely positions
    infState = gameState.deepCopy()

    # Calculate enemies position probability distributions
    for e in self.enemies:
      ePos = gameState.getAgentPosition(e)
      # If there is an accurate position
      if ePos:
        if e not in self.seen:
          # If not yet in seen list then add it
          self.seen.append(e)
        newBeliefs = util.Counter()
        # Set to 1 so that we will get the accurate position
        newBeliefs[ePos] = 1.0
        self.beliefs[e] = newBeliefs
      else:
        # We don't have accurate position so we must infer enemy position
        if e in self.seen:
          # Remove from seen since enemy cannot be seen anymore
          self.seen.remove(e)
        self.elapseTime(e, gameState)
        self.observe(e, gameState)
    #Check distributions
    self.displayDistributionsOverPositions(self.beliefs.values())

    # Creates a new inferred GameState that includes the most likely positions of enemies
    for e in self.enemies:
      guessPos = self.beliefs[e].argMax()
      config = game.Configuration(guessPos, Directions.STOP)
      # Checks if enemy is pacman state by using position on map and which team they are on
      isPac = (infState.isOnRedTeam(e) != infState.isRed(guessPos))
      infState.data.agentStates[e] = game.AgentState(config, isPac)

    # Evaluate strategy that should be used for agents in the current inferred gameState
    self.evalStrat(infState)
    # Run Expectimax to depth 2 since longer will run over time
    act = self.maxFunction(infState, 2)[1]
    return act

  def maxFunction(self, gameState, depth):
    """
    Max function of expectimax to maximize expected utility for our agent
    """
    # End of game or reached depth limit
    if depth == 0 or gameState.isOver():
      return self.evalFunction(gameState), Directions.STOP
    # Get the successor states for our agents possible moves
    acts = self.getActions(gameState, self.index)
    successors = []
    for act in acts:
      successors.append(gameState.generateSuccessor(self.index, act))
    # Get the expected scores of enemy movement
    scores = []
    for succ in successors:
      scores.append(self.expectiFunction(succ, self.enemies[0], depth)[0])
    best = max(scores)
    # Return first best score and action
    for i in range(len(scores)):
      if scores[i] == best:
        return best, acts[i]

  def expectiFunction(self, gameState, enemy, depth):
    """
    This expecti function is used to get the expected result of enemy moves
    and is called for each enemy
    """
    # End of game or reached depth limit
    if depth == 0 or gameState.isOver():
      return self.evalFunction(gameState), Directions.STOP
    # Get the successor states for enemies possible moves
    acts = self.getActions(gameState, enemy)
    successors = []
    for act in acts:
      successors.append(gameState.generateSuccessor(enemy, act))
    # Get the expected scores of enemy movement
    scores = []
    # Call expecti for each enemy otherwise go back to max for our agent
    if enemy == self.enemies[0]:
      # Not final enemy so keep running expecti function
      for succ in successors:
        scores.append(self.expectiFunction(succ, self.enemies[1], depth)[0])
    else:
      for succ in successors:
        scores.append(self.maxFunction(succ, depth - 1)[0])
    # Expected score value
    expScore = sum(scores) / len(scores)
    return expScore, Directions.STOP


  #def evalStrat(self, gameState):


  #def evalFunction(self, gameState):

  def getActions(self, gameState, agent):
    """
    Gets legal actions not including stop for any agent
    """
    actions = gameState.getLegalActions(agent)
    actions.remove(Directions.STOP)
    return actions


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


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)


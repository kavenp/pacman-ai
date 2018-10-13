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

import game, random, util
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
    # Scare timers
    self.enemyScareTime = [gameState.getAgentState(e).scaredTimer for e in self.enemies]
    # List of seen enemies
    self.seen = []
    # List of enemy pacman
    self.enemyPac = []
    # List of enemy ghosts, initialised since enemies start as ghosts
    self.enemyGhost = self.enemies

    # Initialise strategy as one attacker and defender
    self.strats = {}
    self.strats[self.team[0]] = "Attack"
    self.strats[self.team[1]] = "Attack"

    # Flags for agent choice status
    # More detailed than strategy
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

    self.distancer.getMazeDistances()

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
    # Update data
    self.agentPos = gameState.getAgentPosition(self.index)
    self.enemies = self.getOpponents(gameState)
    # Inferred gamestate copy where we will place enemies most likely positions
    infState = gameState.deepCopy()

    # Calculate enemies position probability distributions
    for e in self.enemies:
      # Enemy states
      eState = gameState.getAgentState(e)
      # Update enemy ghost and Pacman lists
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
    #self.displayDistributionsOverPositions(self.beliefs.values())

    # Creates a new inferred GameState that includes the most likely positions of enemies
    for e in self.enemies:
      guessPos = self.beliefs[e].argMax()
      config = game.Configuration(guessPos, Directions.STOP)
      # Checks if enemy is pacman state by using position on map and which team they are on
      isPac = (infState.isOnRedTeam(e) != infState.isRed(guessPos))
      infState.data.agentStates[e] = game.AgentState(config, isPac)

    # Return if only one action available
    possibleActs = self.getActions(infState, self.index)
    if len(possibleActs) == 1:
      return possibleActs[0]

    self.enemyScareTime = [gameState.getAgentState(e).scaredTimer for e in self.enemies]
    # Evaluate strategy that should be used for agents in the current inferred gameState
    self.evalStrat(infState)
    #print self.strats
    # Run Expectimax to depth 2 since longer will run over time
    best, act = self.maxFunction(infState, 2)
    #print best
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
    indexes = []
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
      try:
        successors.append(gameState.generateSuccessor(enemy, act))
      except:
        pass
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

  def evalStrat(self, gameState):
    """
    Evaluation function to determine the agent's strategy
    """
    myState = gameState.getAgentState(self.index)
    enemyScareTime = [gameState.getAgentState(e).scaredTimer for e in self.enemies]
    enemyStates = [gameState.getAgentState(e).isPacman for e in self.enemies]
    enemyPac = enemyStates.count(True)
    enemyPacIndex = [self.enemies[i] for i in range(len(enemyStates)) if enemyStates[i]]
    enemyGhost = 2 - enemyPac
    curStrat = self.strats[self.index]
    teamMateStrat = self.getTeamMateStrat()
    curAtk = (curStrat == "Attack")
    teamAtk = (teamMateStrat == "Attack")
    myAtkNum = len([v for v in self.strats.values() if v == "Attack"])
    myDefNum = 2 - myAtkNum

    # Try to keep the numbers advantage
    if enemyPac == 2 and myDefNum < 2:
      for t in self.team:
        self.strats[t] = "Defense"
    # Defend strategy for the closest teammate
    elif enemyPac == 1:
      #print "ENEMY PACMAN"
      closestTeam = self.getTeamMateClosest(gameState, enemyPacIndex[0])
      if self.index != closestTeam:
        self.strats[closestTeam] = "Defense"
        self.strats[self.index] = "Attack"
      else:
        self.strats[self.index] = "Defense"
        self.strats[self.getTeamMate()] = "Attack"
    else:
      for t in self.team:
        self.strats[t] = "Attack"
    # Full on attack if losing and running out of time or if enemies are scared
    if min(enemyScareTime) > 5 or (gameState.data.timeleft < 250 and self.getScore(gameState) < 0):
      for t in self.team:
        self.strats[t] = "Attack"
    # When we're scared just run
    if myState.scaredTimer > 2:
      self.strats[self.index] = "Run"
      self.strats[self.getTeamMate()] = "Run"


  def evalFunction(self, gameState):
    score = self.getScore(gameState)
    myPos = gameState.getAgentPosition(self.index)
    myState = gameState.getAgentState(self.index)
    enemyScareTime = self.enemyScareTime
    numCarry = gameState.getAgentState(self.index).numCarrying
    enemyPac = []
    enemyGhost = []
    for e in self.enemies:
      eState = gameState.getAgentState(e)
      if eState.isPacman:
        enemyPac.append(e)
      else:
        enemyGhost.append(e)
    # Set carry limits so that agents return after eating enough food
    if score < 5:
      carry = 3
    else:
      carry = 2

    seenNum = len(self.seen)
    food = self.getFood(gameState).asList()
    # Direct orthogonal distance to middle
    #distMid = abs(self.agentPos[0] - (self.mapW/2)) + 1

    distMid = min([self.distancer.getDistance(myPos, (self.mapW/2, i))
         for i in range(self.mapH)
         if (self.mapW/2, i) in self.legalPositions]) + 1

    # Direct orthogonal distance to my side's edge
    if gameState.isOnRedTeam(self.index):
      distHome = abs(self.agentPos[0]) + 1
    else:
      distHome = abs(self.mapW - self.agentPos[0]) + 1

    # Distance from other teammate, split and run
    teamDist = self.distancer.getDistance(myPos, gameState.getAgentPosition(self.getTeamMate()))
    # Positions of the enemies
    enemyPos = [gameState.getAgentPosition(e) for e in self.enemies]
    ghostPos = [gameState.getAgentPosition(ghost) for ghost in enemyGhost]
    pacPos = [gameState.getAgentPosition(pac) for pac in enemyPac]
    # Capsule list
    capsules = self.getCapsules(gameState)
    defCaps = self.getCapsulesYouAreDefending(gameState)

    minScare = min(enemyScareTime)

    # Get minimum distances
    ghostMinD = self.getMinDistance(myPos, ghostPos)
    pacMinD = self.getMinDistance(myPos, pacPos) + 1
    foodMinD = self.getMinDistance(myPos, food) + 1
    capMinD = self.getMinDistance(myPos, capsules)
    defCapMinD = self.getMinDistance(myPos, defCaps)
    enemyMinD = self.getMinDistance(myPos, enemyPos)

    if (minScare != 0) :
      capMinD = capMinD / minScare
    # Flip the signs if they are scared and close enough to catch
    if (minScare > 7 and ghostMinD <= 3) or (not myState.isPacman):
      print "Flipped"
      # Set to minimum ghost distance to 0 if not in danger range
      if ghostMinD >= 4:
        ghostMinD = 0
      ghostMinD *= -10
      seenNum *= -5
    else:
      if pacMinD < ghostMinD:
        ghostMinD = pacMinD
    # Sets if the previous if didn't happen
    if ghostMinD >= 4:
      ghostMinD = 0

    if self.strats[self.index] == "Attack":
      # Reached carry threshold or no more food
      if numCarry > carry or len(food) < 1:
        return - distHome - (100 * numCarry / distMid) + 500 * ghostMinD
      else:
        # Continue attack
        return 2 * score - (8 * numCarry / distMid) - (4 * numCarry / distHome) - 150 * len(food) - 12 * foodMinD \
               - 5000 * len(capsules) - 15 * capMinD + 100 * ghostMinD + 15 * len(self.seen)
    elif self.strats[self.index] == "Defense":
        return -50 * distHome - 1000 * pacMinD - 7 * defCapMinD
    else:
        # Run Away! In "Run" Strategy
        return 4 * score - (7 * numCarry / distMid) - (2 * numCarry / distHome) - 150 * len(food) - 8 * foodMinD - \
               5000 * len(capsules) - 10 * capMinD + 2000 * ghostMinD + 15 * len(self.seen)

  def getTeamMateClosest(self, gameState, enemy):
    """
    Helper that gets the closest team mate to an enemy
    """
    ePos = gameState.getAgentPosition(enemy)
    min = float("inf")
    minTeam = None
    for t in self.team:
      tPos = gameState.getAgentPosition(t)
      dist = self.distancer.getDistance(tPos, ePos)
      if dist < min:
        min = dist
        minTeam = t
    return minTeam

  def getMinDistance(self, pos, posList):
    """
    Gets the minimum distance for a list of positions to pos
    """
    if posList:
      minD = min([self.distancer.getDistance(pos, p) for p in posList])
    else:
      minD = 0
    return minD

  def getTeamMateStrat(self):
    return self.strats[self.getTeamMate()]

  def getTeamMate(self):
    for t in self.team:
      if t != self.index:
        return t

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


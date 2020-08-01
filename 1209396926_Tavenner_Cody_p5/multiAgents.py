# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
	"""
	  A reflex agent chooses an action at each choice point by examining
	  its alternatives via a state evaluation function.

	  The code below is provided as a guide.  You are welcome to change
	  it in any way you see fit, so long as you don't touch our method
	  headers.
	"""


	def getAction(self, gameState):
		"""
		You do not need to change this method, but you're welcome to.

		getAction chooses among the best options according to the evaluation function.

		Just like in the previous project, getAction takes a GameState and returns
		some Directions.X for some X in the set {North, South, West, East, Stop}
		"""
		# Collect legal moves and successor states
		legalMoves = gameState.getLegalActions()

		# Choose one of the best actions
		scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
		bestScore = max(scores)
		bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
		chosenIndex = random.choice(bestIndices) # Pick randomly among the best

		"Add more of your code here if you want to"

		return legalMoves[chosenIndex]

	def evaluationFunction(self, currentGameState, action):
		"""
		Design a better evaluation function here.

		The evaluation function takes in the current and proposed successor
		GameStates (pacman.py) and returns a number, where higher numbers are better.

		The code below extracts some useful information from the state, like the
		remaining food (newFood) and Pacman position after moving (newPos).
		newScaredTimes holds the number of moves that each ghost will remain
		scared because of Pacman having eaten a power pellet.

		Print out these variables to see what you're getting, then combine them
		to create a masterful evaluation function.
		"""
		# Useful information you can extract from a GameState (pacman.py)
		successorGameState = currentGameState.generatePacmanSuccessor(action)
		newPos = successorGameState.getPacmanPosition()
		newFood = successorGameState.getFood()
		newGhostStates = successorGameState.getGhostStates()
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

		"*** YOUR CODE HERE ***"
		fScore = 0

		for x, item in enumerate(newFood):
			for y, food in enumerate(item):
				if not food:
					continue

				fPos = (x, y)
				fDist = util.manhattanDistance(fPos, newPos)
				fScore = fScore + 1.0 / (fDist * fDist)

		closestGhost = 1000
		for ghostState in newGhostStates:
			if ghostState.scaredTimer > 0:
				continue

			gDist = util.manhattanDistance(newPos, ghostState.getPosition())
			closestGhost = min(closestGhost, gDist)

		gScore = 0
		
		if closestGhost < 5:
			gScore = float('-inf') * (5 - closestGhost)

		return successorGameState.getScore() + fScore + gScore

def scoreEvaluationFunction(currentGameState):
	"""
	  This default evaluation function just returns the score of the state.
	  The score is the same one displayed in the Pacman GUI.

	  This evaluation function is meant for use with adversarial search agents
	  (not reflex agents).
	"""
	return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
	"""
	  This class provides some common elements to all of your
	  multi-agent searchers.  Any methods defined here will be available
	  to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

	  You *do not* need to make any changes here, but you can if you want to
	  add functionality to all your adversarial search agents.  Please do not
	  remove anything, however.

	  Note: this is an abstract class: one that should not be instantiated.  It's
	  only partially specified, and designed to be extended.  Agent (game.py)
	  is another abstract class.
	"""

	def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
		self.index = 0 # Pacman is always agent index 0
		self.evaluationFunction = util.lookup(evalFn, globals())
		self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent (question 2)
	"""

	def getAction(self, gameState):
		"""
		  Returns the minimax action from the current gameState using self.depth
		  and self.evaluationFunction.

		  Here are some method calls that might be useful when implementing minimax.

		  gameState.getLegalActions(index):
			Returns a list of legal actions for an agent
			index=0 means Pacman, ghosts are >= 1

		  gameState.generateSuccessor(index, action):
			Returns the successor game state after an agent takes an action

		  gameState.getNumAgents():
			Returns the total number of agents in the game
		"""
		"*** YOUR CODE HERE ***"	    
		def maxindex(gameState, depth, totalGhosts):
			if gameState.isWin() or gameState.isLose() or depth == 0:
				return self.evaluationFunction(gameState)

			v = float('-inf')
			depth+=1
			acts = gameState.getLegalActions(0)

			for action in acts:
				v = max(v, minindex(gameState.generateSuccessor(0, action), depth - 1, 1, totalGhosts))
			return v
	
		def minindex(gameState, depth, index, totalGhosts):
			if gameState.isWin() or gameState.isLose() or depth == 0:
				return self.evaluationFunction(gameState)

			v = float('inf')
			acts = gameState.getLegalActions(index)

			if index == totalGhosts:
				for action in acts:
					v = min(v, maxindex(gameState.generateSuccessor(index, action), depth - 1, totalGhosts))
			else:
				for action in acts:
					v = min(v, minindex(gameState.generateSuccessor(index, action), depth, index + 1, totalGhosts))
			return v

		acts = gameState.getLegalActions()
		totalGhosts = gameState.getNumAgents() - 1
		bestaction = Directions.STOP
		score = float('-inf')

		for action in acts:
			nextState = gameState.generateSuccessor(0, action)
			oldscore = score
			score = max(score, minindex(nextState, self.depth, 1, totalGhosts))
			if score > oldscore:
				bestaction = action

		return bestaction

class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent with alpha-beta pruning (question 3)
	"""

	def getAction(self, gameState):
		"""
		  Returns the minimax action using self.depth and self.evaluationFunction
		"""
		"*** YOUR CODE HERE ***"
		def maxindex(gameState, depth, totalGhosts):
			if gameState.isWin() or gameState.isLose() or depth == 0:
				return self.evaluationFunction(gameState)
			v = float('-inf')
			depth+=1
			acts = gameState.getLegalActions(0)

			for action in acts:
				v = max(v, minindex(gameState.generateSuccessor(0, action), depth - 1, 1, totalGhosts))
			return v
	
		def minindex(gameState, depth, index, totalGhosts):
			if gameState.isWin() or gameState.isLose() or depth == 0:
				return self.evaluationFunction(gameState)
			v = float('inf')

			acts = gameState.getLegalActions(index)

			if index == totalGhosts:
				for action in acts:
					v = min(v, maxindex(gameState.generateSuccessor(index, action), depth - 1, totalGhosts))

			else:
				for action in acts:
					v = min(v, minindex(gameState.generateSuccessor(index, action), depth, index + 1, totalGhosts))

			return v

		acts = gameState.getLegalActions()
		totalGhosts = gameState.getNumAgents() - 1
		bestaction = Directions.STOP
		score = float('-inf')

		for action in acts:
			nextState = gameState.generateSuccessor(0, action)
			oldscore = score
			score = max(score, minindex(nextState, self.depth, 1, totalGhosts))

			if score > oldscore:
				bestaction = action

		return bestaction

class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	  Your expectimax agent (question 4)
	"""

	def getAction(self, gameState):
		"""
		  Returns the expectimax action using self.depth and self.evaluationFunction

		  All ghosts should be modeled as choosing uniformly at random from their
		  legal moves.
		"""
		"*** YOUR CODE HERE ***"
		def expectedValue(gameState, index, depth):
			if gameState.isWin() or gameState.isLose() or depth == 0:
				return self.evaluationFunction(gameState)

			totalGhosts = gameState.getNumAgents() - 1
			legalAct = gameState.getLegalActions(index)
			countActions = len(legalAct)
			value = 0

			for action in legalAct:
				nextState = gameState.generateSuccessor(index, action)
				if (index == totalGhosts):
					value += maxValue(nextState, depth - 1)
				else:
					value += expectedValue(nextState, index + 1, depth)
			return float(value / countActions)

		def maxValue(gameState, depth):
			if gameState.isWin() or gameState.isLose() or depth == 0:
				return self.evaluationFunction(gameState)

			legalAct = gameState.getLegalActions(0)
			bestAction = Directions.STOP
			score = float('-inf')

			for action in legalAct:
				oldScore = score
				nextState = gameState.generateSuccessor(0, action)
				score = max(score, expectedValue(nextState, 1, depth))
			return score

		if gameState.isWin() or gameState.isLose():
			return self.evaluationFunction(gameState)
		legalAct = gameState.getLegalActions(0)
		bestaction = Directions.STOP
		score = float('-inf')

		for action in legalAct:
			nextState = gameState.generateSuccessor(0, action)
			oldScore = score
			score = max(score, expectedValue(nextState, 1, self.depth))
			if score > oldScore:
				bestaction = action

		return bestaction

def betterEvaluationFunction(currentGameState):
	"""
	  Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	  evaluation function (question 5).

	  DESCRIPTION: <write something here so we know what you did>
	"""
	"*** YOUR CODE HERE ***"
	if currentGameState.isWin():
		return float('inf')

	if currentGameState.isLose():
		return float('-inf')

	score = scoreEvaluationFunction(currentGameState)
	newFood = currentGameState.getFood()
	fPos = newFood.asList()
	nextFood = float('inf')

	for pos in fPos:
		fDist = util.manhattanDistance(pos, currentGameState.getPacmanPosition())
		if (fDist < nextFood):
			nextFood = fDist

	totalGhosts = currentGameState.getNumAgents() - 1
	iteration = 1
	nextGhost = float('inf')

	while iteration <= totalGhosts:
		gDist = util.manhattanDistance(currentGameState.getPacmanPosition(), currentGameState.getGhostPosition(iteration))
		nextGhost = min(nextGhost, gDist)
		iteration += 1

	score += max(nextGhost, 4) * 1.5
	score -= nextFood * 1.0

	cLoc = currentGameState.getCapsules()

	score -= 3 * len(fPos)
	score -= 4 * len(cLoc)

	return score
# Abbreviation
better = betterEvaluationFunction


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
import distanceCalculator
import random, time, util
import json
from os.path import exists
import sys
import layout
from game import Directions
import game
from util import nearestPoint

##################
# Game Constants #
##################

# Set TRAINING to True while agents are learning, False if in deployment
# [!] Submit your final team with this set to False!
TRAINING = True

# Name of weights / any agent parameters that should persist between
# games. Should be loaded at the start of any game, training or otherwise
# [!] Replace MY_TEAM with your team name
OFFENSE_WEIGHT_PATH = "offense_Weights_JammyPac.json"
WEIGHT_PATH = "weights_MY_TEAM.json"
TRAINING_WEIGHT_PATH = "training_weights.json"
OFFENSE_WEIGHTS_GHOST_FEATURES = ["distanceToGhost"]
OFFENSE_WEIGHTS_PELLET_FEATURES = [
    "gotCloserToFood",
    "pelletEaten",
]
OFFENSE_WEIGHTS_GENERAL_FEATURES = [
    # "numberPelletsCarried",
    "closerToHome",
]

# OFFENSE_WEIGHTS_OTHER_FEATURES = [
#     "successorScore",
#     # "distanceToFood",
#     # "gotCloserToFood",
#     # "pelletEaten",
#     # "distanceToGhost",
#     "stop",
#     "separationAnxiety",
# ]
OFFENSE_WEIGHTS_ALL_FEATURES = [
    "gotCloserToFood",
    "pelletEaten",
    "distanceToGhost",
    # "numberPelletsCarried",
    "closerToHome",
]

# Any other constants used for your training (learning rate, discount, etc.)
# should be specified here
DISCOUNT_RATE = 0.9
LEARNING_RATE = 0.1
EXPLORATION_RATE = 0.1
# add FRESH_START = False?

#################
# Team creation #
#################


def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first="OffensiveReflexAgent",
    second="DefensiveReflexAgent",
):
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


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

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
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features["successorScore"] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {"successorScore": 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def registerInitialState(self, gameState):

        # built in stuff
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        # record keeping variables, load weights
        self.prevAction = None
        self.prevState = gameState
        self.totalFood = len(self.getFood(gameState).asList())
        self.numOfMoves = 0
        self.width = gameState.data.layout.width
        self.height = gameState.data.layout.height
        self.homeBorders = self.getHomeBorders(gameState)
        if exists(OFFENSE_WEIGHT_PATH):
            self.loadWeights()
        else:
            self.initializeWeights()

    ###### MAIN METHODS ######

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # print("Home Borders:", self.homeBorders)

        # get reward for previous action and update weights
        if TRAINING and self.numOfMoves > 0:
            pelletReward = self.getPelletReward(
                self.prevState, self.prevAction, gameState
            )
            ghostReward = self.getGhostReward(
                self.prevState, self.prevAction, gameState
            )
            generalReward = self.getGeneralReward(
                self.prevState, self.prevAction, gameState
            )

            rewards = {
                "pelletReward": pelletReward,
                "ghostReward": ghostReward,
                "generalReward": generalReward,
            }

            self.updateWeights(self.prevState, self.prevAction, gameState, rewards)

            # store weights into JSON file
            self.storeWeights()
        # get best action
        actions = gameState.getLegalActions(self.index)
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [
            self.evaluate(gameState, a, OFFENSE_WEIGHTS_ALL_FEATURES) for a in actions
        ]
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        # WEIRD BUILT IN LOGIC, Forney says not necessary for now
        # foodLeft = len(self.getFood(gameState).asList())
        # if foodLeft <= 2:
        #     bestDist = 9999
        #     for action in actions:
        #         successor = self.getSuccessor(gameState, action)
        #         pos2 = successor.getAgentPosition(self.index)
        #         dist = self.getMazeDistance(self.start, pos2)
        #         if dist < bestDist:
        #             bestAction = action
        #             bestDist = dist
        #     return bestAction

        # get our action
        if TRAINING and random.random() < EXPLORATION_RATE:
            action = random.choice(actions)
        else:
            action = random.choice(bestActions)

        # store action and state for next weight update
        self.prevAction = action
        self.prevState = gameState
        self.numOfMoves += 1
        return action

    def evaluate(self, gameState, action, set):
        """
        Computes a linear combination of features and feature weights
        """
        # ! Probably can go back to original, just features * weights (ultimately same end result)
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        total = 0
        for key in set:
            total += features[key] * weights[key]
        return total

    def getFeatures(self, gameState, action):

        # record keeping variables
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        food = self.getFood(successor)
        foodList = food.asList()
        prevFood = self.getFood(gameState)
        prevFoodList = prevFood.asList()
        prevPos = gameState.getAgentState(self.index).getPosition()
        nextPos = successor.getAgentState(self.index).getPosition()

        # features["gotCloserToFood"]
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minFoodDistance = min(
                [self.getMazeDistance(nextPos, food) for food in foodList]
            )
        if len(prevFoodList) > 0:
            minPrevFoodDistance = min(
                [self.getMazeDistance(prevPos, food) for food in prevFoodList]
            )
        if len(prevFoodList) > 0 and len(foodList) > 0:
            if minPrevFoodDistance > minFoodDistance:
                features["gotCloserToFood"] = 1.0
            else:
                features["gotCloserToFood"] = 0.0

        # features["pelletEaten"]
        if prevFood.count() > food.count():
            features["pelletEaten"] = 1.0
        else:
            features["pelletEaten"] = 0.0

        features["distanceToGhost"]
        enemies = [
            successor.getAgentState(opponent)
            for opponent in self.getOpponents(successor)
        ]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(ghosts) > 0:
            minGhostDistance = (
                min([self.getMazeDistance(nextPos, a.getPosition()) for a in ghosts])
                + 1
            )

            features["distanceToGhost"] = minGhostDistance / (food.width * food.height)

        # features["numberPelletsCarried"]
        # features["numberPelletsCarried"] = (
        #     successor.getAgentState(self.index).numCarrying / self.totalFood
        # )

        # ! should this be gated w/ if logic? 0 if not carrying?
        # features["closerToHome"]
        minBorderDistancePrev = min(
            [self.getMazeDistance(prevPos, border) for border in self.homeBorders]
        )
        minBorderDistanceCur = min(
            [self.getMazeDistance(nextPos, border) for border in self.homeBorders]
        )

        if (
            minBorderDistanceCur < minBorderDistancePrev
            and gameState.getAgentState(self.index).numCarrying > 0
        ):
            features["closerToHome"] = 1.0
        else:
            features["closerToHome"] = 0.0

        # home = self.start

        # disabled features from BetterBaseline:
        # features["successorScore"] = self.getScore(successor)  # -len(foodList)
        # features["distanceToFood"] = minFoodDistance
        # Can't stop won't stop
        # if action == Directions.STOP:
        #     features["stop"] = 1
        # Safety in numbers! ...even if that number is 2
        # friends = [
        #     successor.getAgentState(friend) for friend in self.getTeam(successor)
        # ]
        # minFriendDistance = (
        #     min([self.getMazeDistance(myPos, a.getPosition()) for a in friends]) + 1
        # )
        # features["separationAnxiety"] = minFriendDistance

        return features

    ###### Rewards ######

    def getGhostReward(self, prevState, prev_action, currentState):
        """
        Design a reward function such that rewards are neither granted too densely nor too sparsely,
        and inform your agent as to what constitutes "good" and "bad" decisions given some decomposition
        of the state. This will, for some actions, depend on taking a "pre" and "post" action snapshot of
        the state to determine the reward, like figuring out if an attacking Pacman died from a move such
        that it was reset to the starting position after running into a defending ghost.
        """
        #

        reward = 0
        prevPos = prevState.getAgentState(self.index).getPosition()
        currentPos = currentState.getAgentState(self.index).getPosition()

        # Negative reward for getting eaten
        if self.getMazeDistance(currentPos, prevPos) > 2:
            reward -= 0.5

        # TODO negative reward for getting closer to ghosts? Negative reward for losing whole game?

        return reward

        # TODO TERMINAL rewards for winning or losing?
        # technically not using prev_action, is it even necessary as a parameter?

    def getGeneralReward(self, prevState, prev_action, currentState):
        """
        Design a reward function such that rewards are neither granted too densely nor too sparsely,
        and inform your agent as to what constitutes "good" and "bad" decisions given some decomposition
        of the state. This will, for some actions, depend on taking a "pre" and "post" action snapshot of
        the state to determine the reward, like figuring out if an attacking Pacman died from a move such
        that it was reset to the starting position after running into a defending ghost.
        """
        #

        reward = 0
        # record keeping
        prevPos = prevState.getAgentState(self.index).getPosition()
        currentPos = currentState.getAgentState(self.index).getPosition()
        carrying = prevState.getAgentState(self.index).numCarrying > 0

        # Reward for getting closer to home (if carrying)
        minBorderPrev = min(
            [self.getMazeDistance(prevPos, border) for border in self.homeBorders]
        )
        minBorderCur = min(
            [self.getMazeDistance(currentPos, border) for border in self.homeBorders]
        )
        print("minBorderPrev", minBorderPrev)
        print("minBorderCur", minBorderCur)
        print("self.homeBorders", self.homeBorders)
        print("currentPos", currentPos)
        if carrying and minBorderCur < minBorderPrev:
            reward += 0.1

        # Reward if pellets returned to base
        print("score", currentState.getScore())
        if prevState.getScore() > currentState.getScore():
            reward += 0.9
        return reward

    def getPelletReward(self, prevState, prev_action, currentState):
        """
        Design a reward function such that rewards are neither granted too densely nor too sparsely,
        and inform your agent as to what constitutes "good" and "bad" decisions given some decomposition
        of the state. This will, for some actions, depend on taking a "pre" and "post" action snapshot of
        the state to determine the reward, like figuring out if an attacking Pacman died from a move such
        that it was reset to the starting position after running into a defending ghost.
        """
        reward = 0

        # get positions & pellet info
        prevPos = prevState.getAgentState(self.index).getPosition()
        currentPos = currentState.getAgentState(self.index).getPosition()
        prevFood = self.getFood(prevState)
        currentFood = self.getFood(currentState)

        # Reward for eating a pellet
        if len(currentFood.asList()) < len(prevFood.asList()):
            reward += 0.1

        # Reward for getting closer to the nearest pellet
        prevMinDistance = min(
            self.getMazeDistance(prevPos, food) for food in prevFood.asList()
        )
        # TODO ValueError: min() arg is an empty sequence solve this
        currentMinDistance = min(
            self.getMazeDistance(currentPos, food) for food in currentFood.asList()
        )
        if currentMinDistance < prevMinDistance:
            reward += 0.01

        # TODO terminal reward for winning?
        return reward

    ###### WEIGHTS ######

    def getWeights(self, gameState, action):
        return self.weights

    def initializeWeights(self):
        self.weights = {}
        for key in OFFENSE_WEIGHTS_ALL_FEATURES:
            self.weights[key] = 0.0

    def loadWeights(self):
        try:
            with open(TRAINING_WEIGHT_PATH, "r") as file:
                self.weights = json.load(file)
        except IOError:
            print("Weights file not found.")

    def storeWeights(self):
        weights_json = json.dumps(self.weights, indent=4)
        with open(TRAINING_WEIGHT_PATH, "w") as outfile:
            outfile.write(weights_json)

    def updateWeights(self, state, action, nextState, rewards):

        # get max Q value for next state:

        actions = nextState.getLegalActions(self.index)
        features = self.getFeatures(state, action)
        print("features", features)

        # TODO cleaner way to do this bc this is a mess!!

        # update Pellet Weights
        pelletValues = [
            self.evaluate(nextState, a, OFFENSE_WEIGHTS_PELLET_FEATURES)
            for a in actions
        ]
        maxPelletValue = max(pelletValues)

        pelletDifference = (
            rewards["pelletReward"] + DISCOUNT_RATE * maxPelletValue
        ) - self.evaluate(state, action, OFFENSE_WEIGHTS_PELLET_FEATURES)

        for feature in OFFENSE_WEIGHTS_PELLET_FEATURES:
            self.weights[feature] += (
                LEARNING_RATE * pelletDifference * features[feature]
            )

        # update Ghost Weights
        ghostValues = [
            self.evaluate(nextState, a, OFFENSE_WEIGHTS_GHOST_FEATURES) for a in actions
        ]
        maxGhostValue = max(ghostValues)

        ghostDifference = (
            rewards["ghostReward"] + DISCOUNT_RATE * maxGhostValue
        ) - self.evaluate(state, action, OFFENSE_WEIGHTS_GHOST_FEATURES)

        for feature in OFFENSE_WEIGHTS_GHOST_FEATURES:
            self.weights[feature] += LEARNING_RATE * ghostDifference * features[feature]

        # update General Weights
        generalValues = [
            self.evaluate(nextState, a, OFFENSE_WEIGHTS_GENERAL_FEATURES)
            for a in actions
        ]
        maxGeneralValue = max(generalValues)

        generalDifference = (
            rewards["generalReward"] + DISCOUNT_RATE * maxGeneralValue
        ) - self.evaluate(state, action, OFFENSE_WEIGHTS_GENERAL_FEATURES)

        for feature in OFFENSE_WEIGHTS_GENERAL_FEATURES:
            self.weights[feature] += (
                LEARNING_RATE * generalDifference * features[feature]
            )

    ###### OTHER HELPERS ######

    def getHomeBorders(self, gameState):
        """
        Get Home Border Locs
        """
        walls = gameState.getWalls().asList()
        if self.red:
            homeBorderX = self.width // 2 - 1
        else:
            homeBorderX = self.width // 2
        allBorders = [(homeBorderX, y) for y in range(self.height)]
        bordersMinusWalls = [loc for loc in allBorders if loc not in walls]
        return bordersMinusWalls


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features["onDefense"] = 1
        if myState.isPacman:
            features["onDefense"] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features["numInvaders"] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features["invaderDistance"] = min(dists)

        if action == Directions.STOP:
            features["stop"] = 1
        rev = Directions.REVERSE[
            gameState.getAgentState(self.index).configuration.direction
        ]
        if action == rev:
            features["reverse"] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            "numInvaders": -1000,
            "onDefense": 100,
            "invaderDistance": -10,
            "stop": -100,
            "reverse": -2,
        }


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

        """
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    """
        CaptureAgent.registerInitialState(self, gameState)

        """
    Your initialization code goes here, if you need any.
    """

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        """
    You should change this in your own agent.
    """

        return random.choice(actions)

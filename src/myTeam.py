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
# MARK: Constants
##################

# TODO Submit your final team with this set to False!
TRAINING = True

# Name of weights / any agent parameters that should persist between
# games. Should be loaded at the start of any game, training or otherwise
# [!] Replace MY_TEAM with your team name
OFFENSE_WEIGHT_PATH = "offense_weights.json"
DEFENSE_WEIGHT_PATH = "defense_weights.json"
PELLET = "pellets"
GHOST = "ghosts"
GENERAL = "general"
OFFENSE_FEATURE_SECTIONS = [PELLET, GHOST, GENERAL]
OFFENSE_FEATURES_GHOST = ["distanceToGhost"]
OFFENSE_PELLET_FEATURES = [
    "gotCloserToFood",
    "pelletEaten",
]
OFFENSE_GENERAL_FEATURES = [
    "closerToHome",
]

DEFENSE_WEIGHTS_HUNT_FEATURES = ["gotCloserToInvader", "ateInvader"]
DEFENSE_WEIGHTS_STAYDEF_FEATURES = ["onDefense", "leavingStart", "coveringGround"]


DEFENSE_WEIGHTS_HUNT_FEATURES = ["gotCloserToInvader", "ateInvader", "enemyPowered"]
DEFENSE_WEIGHTS_STAYDEF_FEATURES = [
    "onDefense",
    "leavingStart",
    "coveringGround",
    "stopped",
]

DEFENSE_WEIGHTS_ALL_FEATURES = [
    "gotCloserToInvader",
    "ateInvader",
    "enemyPowered",
    "onDefense",
    "leavingStart",
    "coveringGround",
    "stopped",
]

# Any other constants used for your training (learning rate, discount, etc.)
# should be specified here
DISCOUNT_RATE = 0.9
LEARNING_RATE = 0.2
EXPLORATION_RATE = 0.1
FRESH_START = False

#################
# MARK: TEAM CREATION
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


# MARK: REFLEX AGENT
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


# MARK: OFFENSE AGENT
class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    # MARK: INITIALIZE
    def registerInitialState(self, gameState):

        # built in stuff
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        # record keeping variables
        self.prevAction = None
        self.prevState = gameState
        self.totalFood = len(self.getFood(gameState).asList())
        self.numOfMoves = 0
        self.width = gameState.data.layout.width
        self.height = gameState.data.layout.height
        self.homeBorders = self.getHomeBorders(gameState)
        if exists(OFFENSE_WEIGHT_PATH) and not FRESH_START:
            print("Loading weights from file")
            self.loadWeights()
        else:
            print("Initializing weights")
            self.initializeWeights()

    ###### MAIN METHODS ######
    # MARK: CHOOSE ACTION
    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # get reward for previous action and update weights
        if TRAINING and self.numOfMoves > 0:
            rewards = self.getRewards(self.prevState, self.prevAction, gameState)
            self.updateWeights(self.prevState, self.prevAction, gameState, rewards)

        # GET BEST ACTION
        actions = gameState.getLegalActions(self.index)
        # You can profile your evaluation time by uncommenting these lines
        start = time.time()
        values = [self.evaluatePooling(gameState, a) for a in actions]
        print("eval time for agent %d: %.4f" % (self.index, time.time() - start))
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        if TRAINING and random.random() < EXPLORATION_RATE:
            action = random.choice(actions)
        else:
            action = random.choice(bestActions)

        # record keeping for next reward calculation
        self.prevAction = action
        self.prevState = gameState
        self.numOfMoves += 1

        # WEIRD BUILT IN LOGIC, Forney says not necessary for now
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
        return action

    # MARK: EVALUATE
    def evaluatePooling(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        total = 0
        for key in OFFENSE_FEATURE_SECTIONS:
            for feature in features[key]:
                total += features[key][feature] * weights[key][feature]
        return total

    def evaluate(self, gameState, action, featureKey):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)[featureKey]
        weights = self.getWeights(gameState, action)[featureKey]
        return features * weights

    # MARK: GETFEATURES
    def getFeatures(self, gameState, action):

        # record keeping variables
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # set up Features dictionary
        for key in OFFENSE_FEATURE_SECTIONS:
            features[key] = util.Counter()

        # MARK: pellet features
        # pellet feature variables
        food = self.getFood(successor)
        foodList = food.asList()
        prevFood = self.getFood(gameState)
        prevFoodList = prevFood.asList()
        prevPos = gameState.getAgentState(self.index).getPosition()
        nextPos = successor.getAgentState(self.index).getPosition()
        # features[PELLET]["gotCloserToFood"]
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
                features[PELLET]["gotCloserToFood"] = 1.0
            else:
                features[PELLET]["gotCloserToFood"] = 0.0
        # features[PELLET]["pelletEaten"]
        if prevFood.count() > food.count():
            features[PELLET]["pelletEaten"] = 1.0
        else:
            features[PELLET]["pelletEaten"] = 0.0

        # MARK: ghost features
        # features[GHOST]["distanceToGhost"]
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
            features[GHOST]["distanceToGhost"] = minGhostDistance / (
                self.width * self.height
            )
        else:
            features[GHOST]["distanceToGhost"] = 0.0

        # GENERAL FEATURES
        # features[GENERAL]["closerToHome"]
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
            features[GENERAL]["closerToHome"] = 1.0
        else:
            features[GENERAL]["closerToHome"] = 0.0

        # WORKS IN PROGRESS
        # features["numberPelletsCarried"]
        # features["numberPelletsCarried"] = (
        #     successor.getAgentState(self.index).numCarrying / self.totalFood
        # )

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

    # MARK: REWARDS

    def getRewards(self, prevState, prev_action, currentState):
        pelletReward = self.getPelletReward(prevState, prev_action, currentState)
        ghostReward = self.getGhostReward(prevState, prev_action, currentState)
        generalReward = self.getGeneralReward(prevState, prev_action, currentState)
        return {
            PELLET: pelletReward,
            GHOST: ghostReward,
            GENERAL: generalReward,
        }

    def getGhostReward(self, prevState, prev_action, currentState):
        """
        Design a reward function such that rewards are neither granted too densely nor too sparsely,
        and inform your agent as to what constitutes "good" and "bad" decisions given some decomposition
        of the state. This will, for some actions, depend on taking a "pre" and "post" action snapshot of
        the state to determine the reward, like figuring out if an attacking Pacman died from a move such
        that it was reset to the starting position after running into a defending ghost.
        """
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
        if carrying and minBorderCur < minBorderPrev:
            reward += 0.1

        # Reward if pellets returned to base
        print("score", currentState.getScore())
        if self.red and currentState.getScore() > prevState.getScore():
            reward += 0.9
        if not self.red and prevState.getScore() > currentState.getScore():
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

    # MARK: WEIGHTS

    def getWeights(self, gameState, action):
        return self.weights

    def initializeWeights(self):
        self.weights = util.Counter()
        for key in OFFENSE_FEATURE_SECTIONS:
            self.weights[key] = util.Counter()
        for feature in OFFENSE_PELLET_FEATURES:
            self.weights[PELLET][feature] = 0.0
        for feature in OFFENSE_FEATURES_GHOST:
            self.weights[GHOST][feature] = 0.0
        for feature in OFFENSE_GENERAL_FEATURES:
            self.weights[GENERAL][feature] = 0.0

    def loadWeights(self):
        try:
            with open(OFFENSE_WEIGHT_PATH, "r") as file:
                self.weights = json.load(file)
        except IOError:
            print("Weights file not found.")

    def storeWeights(self):
        weights_json = json.dumps(self.weights, indent=4)
        with open(OFFENSE_WEIGHT_PATH, "w") as outfile:
            outfile.write(weights_json)

    # MARK: Update Weights
    def updateWeights(self, state, action, nextState, rewards):
        actions = nextState.getLegalActions(self.index)
        features = self.getFeatures(state, action)
        print("features", features)
        print("weights", self.weights)
        for key in OFFENSE_FEATURE_SECTIONS:
            print("key", key)
            print("rewards", rewards)
            values = [self.evaluate(nextState, action, key) for action in actions]
            maxValue = max(values)
            difference = (rewards[key] + DISCOUNT_RATE * maxValue) - self.evaluate(
                state, action, key
            )
            for feature in self.weights[key]:
                self.weights[key][feature] += (
                    LEARNING_RATE * difference * features[key][feature]
                )

    # MARK: HELPERS
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

    def final(self, gameState):
        if TRAINING:
            print("Storing weights as game is over.")
            self.storeWeights()


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def registerInitialState(self, gameState):

        # built in stuff
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        # record keeping variables, load weights
        self.prevAction = None
        self.prevState = gameState
        self.numOfMoves = 0
        if exists(DEFENSE_WEIGHT_PATH):
            self.loadWeights()
        else:
            self.initializeWeights()

        self.coverageMap = util.Counter()

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """

        # get reward for previous action and update weights
        if TRAINING and self.numOfMoves > 0:
            huntReward = self.getHuntReward(self.prevState, self.prevAction, gameState)
            onDefReward = self.getDefendingReward(
                self.prevState, self.prevAction, gameState
            )
            rewards = {"huntReward": huntReward, "onDefReward": onDefReward}

            self.updateWeights(self.prevState, self.prevAction, gameState, rewards)

        # get best action
        actions = gameState.getLegalActions(self.index)
        # You can profile your evaluation time by uncommenting these lines
        start = time.time()
        values = [
            self.evaluate(gameState, a, DEFENSE_WEIGHTS_ALL_FEATURES) for a in actions
        ]
        print("eval time for agent %d: %.4f" % (self.index, time.time() - start))
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

        # update coverage map
        newPos = gameState.getAgentPosition(self.index)
        self.coverageMap[newPos] = 0
        for pos, lastVisit in self.coverageMap.items():
            self.coverageMap[pos] = lastVisit + 1

        # store action and state for next weight update
        self.prevAction = action
        self.prevState = gameState
        self.numOfMoves += 1
        return action

    def evaluate(self, gameState, action, set):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        total = 0
        for key in set:
            total += features[key] * weights[key]
        return total

    def getHuntReward(self, prevState, prev_action, currentState):
        reward = 0
        myPos = currentState.getAgentState(self.index).getPosition()
        prevPos = prevState.getAgentState(self.index).getPosition()

        # Computes distance to invaders we can see
        enemies = [
            currentState.getAgentState(i) for i in self.getOpponents(currentState)
        ]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        prevEnemies = [prevState.getAgentState(i) for i in self.getOpponents(prevState)]
        prevInvaders = [
            a for a in prevEnemies if a.isPacman and a.getPosition() != None
        ]
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            closestInvaderDistance = min(dists)
        if len(prevInvaders) > 0:
            prevDists = [
                self.getMazeDistance(prevPos, a.getPosition()) for a in prevInvaders
            ]
            prevClosestInvaderDistance = min(prevDists)

        if (
            len(invaders) > 0
            and len(prevInvaders) > 0
            and closestInvaderDistance < prevClosestInvaderDistance
        ):
            reward += 0.09

        if len(prevInvaders) > 0:
            if prevClosestInvaderDistance <= 1 and len(invaders) < len(prevInvaders):
                reward += 0.9
            # if power pellet invert reward TODO TODO
        return reward

    def getDefendingReward(self, prevState, prev_action, currentState):
        reward = 0
        myPos = currentState.getAgentState(self.index).getPosition()
        prevPos = prevState.getAgentState(self.index).getPosition()
        myPosAsFloat = (float(myPos[0]), float(myPos[1]))
        if currentState.getAgentState(self.index).isPacman:
            reward += -0.012
        if (
            self.getMazeDistance(myPos, self.start)
            > self.getMazeDistance(prevPos, self.start)
            and self.numOfMoves < 10
        ):
            reward += 0.001
        if myPos not in self.coverageMap.keys():
            reward += 0.01
        elif self.coverageMap[myPos] >= 12:
            reward += 0.009
        return reward

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        prevPos = gameState.getAgentState(self.index).getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features["onDefense"] = 1.0
        if myState.isPacman:
            features["onDefense"] = 0.0

        features["leavingStart"] = 0.0
        if (
            self.getMazeDistance(myPos, self.start)
            > self.getMazeDistance(prevPos, self.start)
            and self.numOfMoves < 10
        ):
            features["leavingStart"] = 1.0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        prevEnemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        prevInvaders = [
            a for a in prevEnemies if a.isPacman and a.getPosition() != None
        ]
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            closestInvaderDistance = min(dists)
        if len(prevInvaders) > 0:
            prevDists = [
                self.getMazeDistance(prevPos, a.getPosition()) for a in prevInvaders
            ]
            prevClosestInvaderDistance = min(prevDists)

        if (
            len(invaders) > 0
            and len(prevInvaders) > 0
            and closestInvaderDistance < prevClosestInvaderDistance
        ):
            features["gotCloserToInvader"] = 1.0
        else:
            features["gotCloserToInvader"] = 0.0

        features["ateInvader"] = 0.0
        if len(prevInvaders) > 0:
            if prevClosestInvaderDistance <= 1 and len(invaders) < len(prevInvaders):
                features["ateInvader"] = 1.0

        if myPos not in self.coverageMap.keys() or self.coverageMap[myPos] >= 6:
            features["coveringGround"] = 1.0
        else:
            features["coveringGround"] = 0.0

        # if action == Directions.STOP:
        #     features["stop"] = 1
        # rev = Directions.REVERSE[
        #     gameState.getAgentState(self.index).configuration.direction
        # ]
        # if action == rev:
        #     features["reverse"] = 1

        return features

    def getWeights(self, gameState, action):
        return self.weights

    def updateWeights(self, state, action, nextState, rewards):
        actions = nextState.getLegalActions(self.index)
        features = self.getFeatures(state, action)

        huntValues = [
            self.evaluate(nextState, a, DEFENSE_WEIGHTS_HUNT_FEATURES) for a in actions
        ]
        maxHuntValue = max(huntValues)

        onDefValues = [
            self.evaluate(nextState, a, DEFENSE_WEIGHTS_STAYDEF_FEATURES)
            for a in actions
        ]
        maxOnDefValue = max(onDefValues)

        huntDifference = (
            rewards["huntReward"] + DISCOUNT_RATE * maxHuntValue
        ) - self.evaluate(state, action, DEFENSE_WEIGHTS_HUNT_FEATURES)

        for feature in DEFENSE_WEIGHTS_HUNT_FEATURES:
            self.weights[feature] += LEARNING_RATE * huntDifference * features[feature]

        onDefDifference = (
            rewards["onDefReward"] + DISCOUNT_RATE * maxOnDefValue
        ) - self.evaluate(state, action, DEFENSE_WEIGHTS_STAYDEF_FEATURES)

        for feature in DEFENSE_WEIGHTS_STAYDEF_FEATURES:
            self.weights[feature] += LEARNING_RATE * onDefDifference * features[feature]

    # TODO TODO TODO change paths to new ones TODO TODO TODO
    def storeWeights(self):
        weights_json = json.dumps(self.weights, indent=4)
        with open(DEFENSE_WEIGHT_PATH, "w") as outfile:
            outfile.write(weights_json)

    def loadWeights(self):
        try:
            with open(DEFENSE_WEIGHT_PATH, "r") as file:
                self.weights = json.load(file)
        except IOError:
            print("Weights file not found.")

    def initializeWeights(self):
        self.weights = {}
        for key in DEFENSE_WEIGHTS_ALL_FEATURES:
            self.weights[key] = 0.0

    def final(self, gameState):
        if TRAINING:
            # print("Storing weights as game is over.")
            self.storeWeights()

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
TRAINING_WEIGHT_PATH = "training_weights.json"
PELLET = "pellets"
GHOST = "ghosts"
GENERAL = "general"
NEAR_PELLETS = "near_pellets"
OFFENSE_FEATURE_SECTIONS = [PELLET, GHOST, GENERAL, NEAR_PELLETS]
OFFENSE_FEATURES_GHOST = ["death", "distanceToGhost"]  # "ghostCloser"
OFFENSE_PELLET_FEATURES = [
    "gotCloserToFood",
    "pelletEaten",
]
OFFENSE_NEAR_PELLET_FEATURES = ["closerToNearFood", "safeClosePellet"]
OFFENSE_GENERAL_FEATURES = [
    "closerToHome",
]
# OFFENSE_FEATURES_TODO = [
#     "successorScore",
#     # "distanceToFood",
#     # "gotCloserToFood",
#     # "pelletEaten",
#     # "distanceToGhost",
#     "stop",
#     "separationAnxiety",
# ]
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
        self.deaths = 0

        # record keeping variables
        self.prevAction = None
        self.prevState = gameState
        self.totalFood = len(self.getFood(gameState).asList())
        self.numOfMoves = 0
        self.width = gameState.data.layout.width
        self.height = gameState.data.layout.height
        self.homeBorders = self.getHomeBorders(gameState)
        if exists(TRAINING_WEIGHT_PATH) and not FRESH_START:
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
        # start = time.time()
        values = [self.evaluatePooling(gameState, a) for a in actions]
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
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
        print("move no: ", self.numOfMoves)
        print("=====")

        return action

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

    # MARK: EVALUATE
    def evaluatePooling(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        total = 0
        print("features", features)
        print("=============")
        print("weights", weights)
        print("=============")

        for key in OFFENSE_FEATURE_SECTIONS:
            for feature in features[key]:
                total += features[key][feature] * weights[key][feature]
        return total

    def evaluate(self, gameState, action, featureKey):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)[featureKey]
        # print("features", features)
        weights = self.getWeights(gameState, action)[featureKey]
        # print("weights", weights)
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

        # features[GHOST]["death"]
        if self.getMazeDistance(nextPos, prevPos) > 2:
            features[GHOST]["death"] = 1.0
            print("fuck death feature")
            print("@@@@@@@@@@@@@@@")
        else:
            features[GHOST]["death"] = 0.0

        # get current enemies
        enemiesCurr = [
            successor.getAgentState(opponent)
            for opponent in self.getOpponents(successor)
        ]
        ghostsCurr = [
            a for a in enemiesCurr if not a.isPacman and a.getPosition() != None
        ]
        if len(ghostsCurr) > 0:
            minGhostDistanceCurr = (
                min(
                    [self.getMazeDistance(nextPos, a.getPosition()) for a in ghostsCurr]
                )
                + 1
            )
            features[GHOST]["distanceToGhost"] = (
                self.width * self.height - minGhostDistanceCurr
            )

        # get previous enemies
        enemiesPrev = [
            gameState.getAgentState(opponent)
            for opponent in self.getOpponents(gameState)
        ]
        ghostPrev = [
            a for a in enemiesPrev if not a.isPacman and a.getPosition() != None
        ]
        if len(ghostPrev) > 0:
            minGhostDistancePrev = (
                min([self.getMazeDistance(prevPos, a.getPosition()) for a in ghostPrev])
                + 1
            )

        # features[GHOST]["ghostCloser"] = 0.0
        # if (
        #     len(ghostPrev)
        #     and len(ghostsCurr)
        #     and minGhostDistanceCurr - minGhostDistancePrev > 2
        # ):
        #     print("minGhostDistanceCurr", minGhostDistanceCurr)
        #     print("minGhostDistancePrev", minGhostDistancePrev)
        #     minGhostDistanceCurr = 1
        #     print("minGhostDistanceCurrRevised", minGhostDistanceCurr)

        # if len(ghostPrev) and len(ghostsCurr):
        #     if minGhostDistanceCurr < minGhostDistancePrev:
        #         features[GHOST]["ghostCloser"] = 1.0

        #     features[GHOST]["distanceToGhost"] = minGhostDistance / (
        #         self.width * self.height
        #     )
        # else:
        #     features[GHOST]["distanceToGhost"] = 0.0

        # features[NEAR_PELLETS]["safeClosePellet"]
        noisies = []
        for enemyIndex in self.getOpponents(gameState):
            noisyDistance = gameState.getAgentDistances()[enemyIndex]
            noisies.append(noisyDistance)
        features[NEAR_PELLETS]["safeClosePellet"] = 0.0
        for enemy in enemiesCurr:
            if not enemy.isPacman and enemy.getPosition() == None:
                noisies = []
                for enemyIndex in self.getOpponents(gameState):
                    noisyDistance = successor.getAgentDistances()[enemyIndex]
                    noisies.append(noisyDistance)
                if min(noisies) - 6 > minFoodDistance:
                    features[NEAR_PELLETS]["safeClosePellet"] = 1.0
        # features[NEAR_PELLETS]["closerToNearFood"]
        if len(prevFoodList) > 0 and len(foodList) > 0:
            if minPrevFoodDistance > minFoodDistance and minFoodDistance < 7:
                features[NEAR_PELLETS]["closerToNearFood"] = 1.0
            else:
                features[NEAR_PELLETS]["closerToNearFood"] = 0.0
        # MARK: general features
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
        nearReward = self.getNearPelletReward(prevState, prev_action, currentState)
        return {
            PELLET: pelletReward,
            GHOST: ghostReward,
            GENERAL: generalReward,
            NEAR_PELLETS: nearReward,
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

        enemiesPrev = [
            prevState.getAgentState(opponent)
            for opponent in self.getOpponents(prevState)
        ]
        enemiesCurr = [
            currentState.getAgentState(opponent)
            for opponent in self.getOpponents(currentState)
        ]

        ghostsCurr = [
            a for a in enemiesCurr if not a.isPacman and a.getPosition() != None
        ]
        ghostsPrev = [
            a for a in enemiesPrev if not a.isPacman and a.getPosition() != None
        ]
        if len(ghostsCurr) > 0:
            minGhostDistanceCurr = (
                min(
                    [
                        self.getMazeDistance(currentPos, a.getPosition())
                        for a in ghostsCurr
                    ]
                )
                + 1
            )
        if len(ghostsPrev) > 0:
            minGhostDistancePrev = (
                min(
                    [self.getMazeDistance(prevPos, a.getPosition()) for a in ghostsPrev]
                )
                + 1
            )
        if len(ghostsCurr) > 0 and len(ghostsPrev) > 0:
            if minGhostDistanceCurr - minGhostDistancePrev > 2:
                # print("minGhostDistanceCurr", minGhostDistanceCurr)
                # print("minGhostDistancePrev", minGhostDistancePrev)
                minGhostDistanceCurr = 1
                # print("minGhostDistanceCurrRevised", minGhostDistanceCurr)
        if len(ghostsPrev) > 0 and len(ghostsCurr) > 0:
            if minGhostDistanceCurr < minGhostDistancePrev:
                # TODO never happening!
                reward -= 0.08
        if len(ghostsPrev) and minGhostDistancePrev < 3:
            reward -= 0.05

        # Negative reward for getting eaten
        if self.getMazeDistance(currentPos, prevPos) > 2:
            self.deaths += 1
            print("blarg?")
            reward -= 0.5
        # print("ghost reward", reward)
        return reward

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
        # print("score", currentState.getScore())
        if self.red and currentState.getScore() > prevState.getScore():
            reward += 0.3
        if not self.red and prevState.getScore() > currentState.getScore():
            reward += 0.3
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

    # MARK: near pellets

    def getNearPelletReward(self, prevState, prev_action, currentState):
        reward = 0
        food = self.getFood(currentState)
        foodList = food.asList()
        prevFood = self.getFood(prevState)
        prevFoodList = prevFood.asList()
        prevPos = prevState.getAgentState(self.index).getPosition()
        nextPos = currentState.getAgentState(self.index).getPosition()

        # features[PELLET]["gotCloserToFood"]
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minFoodDistance = min(
                [self.getMazeDistance(nextPos, food) for food in foodList]
            )
        if len(prevFoodList) > 0:
            minPrevFoodDistance = min(
                [self.getMazeDistance(prevPos, food) for food in prevFoodList]
            )
        if minFoodDistance > minPrevFoodDistance and minFoodDistance < 7:
            reward += 0.4

        noisies = []
        for enemyIndex in self.getOpponents(currentState):
            noisyDistance = currentState.getAgentDistances()[enemyIndex]
            noisies.append(noisyDistance)
        enemies = [
            currentState.getAgentState(opponent)
            for opponent in self.getOpponents(currentState)
        ]
        for enemy in enemies:
            if not enemy.isPacman and enemy.getPosition() == None:
                noisies = []
                for enemyIndex in self.getOpponents(currentState):
                    noisyDistance = currentState.getAgentDistances()[enemyIndex]
                    noisies.append(noisyDistance)
                if min(noisies) - 5 > minFoodDistance and reward == 0.4:
                    # print("near pellets", reward)
                    reward += 0.6
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
        for feature in OFFENSE_NEAR_PELLET_FEATURES:
            self.weights[NEAR_PELLETS][feature] = 0.0

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

    # MARK: Update Weights
    def updateWeights(self, state, action, nextState, rewards):
        actions = nextState.getLegalActions(self.index)
        features = self.getFeatures(state, action)
        # print("=====features======")
        # print("features", features)
        # print("=====weights======")
        # print("weights", self.weights)
        for key in OFFENSE_FEATURE_SECTIONS:
            # print("key", key)
            # print("rewards", rewards)
            values = [self.evaluate(nextState, a, key) for a in actions]
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
        print("Storing weights for game over.")
        self.storeWeights()


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

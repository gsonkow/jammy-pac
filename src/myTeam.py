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
OFFENSE_WEIGHTS_GHOST_PATH = "offense_Weights_Ghost.json"
OFFENSE_WEIGHTS_PELLET_PATH = "offense_Weights_Pellet.json"
OFFENSE_WEIGHTS_OTHER_PATH = "offense_Weights_Other.json"
OFFENSE_WEIGHTS_GHOST_FEATURES = ["distanceToGhost"]
OFFENSE_WEIGHTS_PELLET_FEATURES = [
    "gotCloserToFood",
    "pelletEaten",
]
DEFENSE_WEIGHTS_HUNT_FEATURES = ["gotCloserToInvader", "ateInvader"]
DEFENSE_WEIGHTS_STAYDEF_FEATURES = ["onDefense"]

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
]
DEFENSE_WEIGHTS_ALL_FEATURES = [
    "gotCloserToInvader", 
    "ateInvader",
    "onDefense"
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
# ! WEIGHT questions
# - having trouble with inialization logic. If training...  training, do we initialize weights here?

# FONERY - where do we initialize weights?
# ! how to handle defense vs offense in the JSON
# ! where is update taking place?
# ! do we need that "if ghost near, don't eat pellet" logic?


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
        self.numOfMoves = 0
        if exists(OFFENSE_WEIGHT_PATH):
            self.loadWeights()
        else:
            self.initializeWeights()

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """

        # get reward for previous action and update weights
        if TRAINING and self.numOfMoves > 0:
            pelletReward = self.getPelletReward(
                self.prevState, self.prevAction, gameState
            )
            ghostReward = self.getGhostReward(
                self.prevState, self.prevAction, gameState
            )
            rewards = {"pelletReward": pelletReward, "ghostReward": ghostReward}

            self.updateWeights(self.prevState, self.prevAction, gameState, rewards)

            # store weights into JSON file
            if gameState.isOver():
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
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        total = 0
        for key in set:
            total += features[key] * weights[key]
        return total

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
        currentMinDistance = min(
            self.getMazeDistance(currentPos, food) for food in currentFood.asList()
        )
        if currentMinDistance < prevMinDistance:
            reward += 0.01

        # TODO terminal reward for winning?
        return reward

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        food = self.getFood(successor)
        foodList = food.asList()
        prevFood = self.getFood(gameState)
        prevFoodList = prevFood.asList()
        prevPos = gameState.getAgentState(self.index).getPosition()
        nextPos = successor.getAgentState(self.index).getPosition()

        # Compute distance to the nearest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minFoodDistance = min(
                [self.getMazeDistance(nextPos, food) for food in foodList]
            )

        # Compute distance to the nearest food in prev state
        if (
            len(prevFoodList) > 0
        ):  # This should always be True,  but better safe than sorry
            minPrevFoodDistance = min(
                [self.getMazeDistance(prevPos, food) for food in prevFoodList]
            )

        if len(prevFoodList) > 0 and len(foodList) > 0:
            if minPrevFoodDistance > minFoodDistance:
                features["gotCloserToFood"] = 1.0
            else:
                features["gotCloserToFood"] = 0.0

        # encode if pellet eaten by action
        if prevFood.count() > food.count():
            features["pelletEaten"] = 1.0
        else:
            features["pelletEaten"] = 0.0

        # disabled features from BetterBaseline:
        # features["successorScore"] = -len(foodList)  # self.getScore(successor)
        # features["distanceToFood"] = minFoodDistance

        # Better baselines will avoid defenders!
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
            features["distanceToGhost"] = minGhostDistance

        # Can't stop won't stop
        # if action == Directions.STOP:
        #     features["stop"] = 1
        # print("=========features===============")
        # print(features)
        # print("nextPos: ", nextPos)
        # print("action: ", action)
        # Safety in numbers! ...even if that number is 2
        # friends = [
        #     successor.getAgentState(friend) for friend in self.getTeam(successor)
        # ]
        # minFriendDistance = (
        #     min([self.getMazeDistance(myPos, a.getPosition()) for a in friends]) + 1
        # )
        # features["separationAnxiety"] = minFriendDistance

        return features

    def getWeights(self, gameState, action):
        return self.weights

    def updateWeights(self, state, action, nextState, rewards):

        # get max Q value for next state:

        actions = nextState.getLegalActions(self.index)
        features = self.getFeatures(state, action)

        pelletValues = [
            self.evaluate(nextState, a, OFFENSE_WEIGHTS_PELLET_FEATURES)
            for a in actions
        ]
        maxPelletValue = max(pelletValues)

        ghostValues = [
            self.evaluate(nextState, a, OFFENSE_WEIGHTS_GHOST_FEATURES) for a in actions
        ]
        maxGhostValue = max(ghostValues)

        pelletDifference = (
            rewards["pelletReward"] + DISCOUNT_RATE * maxPelletValue
        ) - self.evaluate(state, action, OFFENSE_WEIGHTS_PELLET_FEATURES)

        # update weights
        for feature in OFFENSE_WEIGHTS_PELLET_FEATURES:
            self.weights[feature] += (
                LEARNING_RATE * pelletDifference * features[feature]
            )

        ghostDifference = (
            rewards["ghostReward"] + DISCOUNT_RATE * maxGhostValue
        ) - self.evaluate(state, action, OFFENSE_WEIGHTS_GHOST_FEATURES)
        # update weights
        for feature in OFFENSE_WEIGHTS_GHOST_FEATURES:
            self.weights[feature] += LEARNING_RATE * ghostDifference * features[feature]

    def storeWeights(self):
        weights_json = json.dumps(self.weights, indent=4)
        with open(TRAINING_WEIGHT_PATH, "w") as outfile:
            outfile.write(weights_json)

    def loadWeights(self):
        try:
            with open(TRAINING_WEIGHT_PATH, "r") as file:
                self.weights = json.load(file)
        except IOError:
            print("Weights file not found.")

    def initializeWeights(self):
        self.weights = {}
        for key in OFFENSE_WEIGHTS_ALL_FEATURES:
            self.weights[key] = 0.0


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
        if exists(OFFENSE_WEIGHT_PATH):
            self.loadWeights()
        else:
            self.initializeWeights()

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """

        # get reward for previous action and update weights
        if TRAINING and self.numOfMoves > 0:
            huntReward = self.getHuntReward(
                self.prevState, self.prevAction, gameState
            )
            onDefReward = self.getDefendingReward(
                self.prevState, self.prevAction, gameState
            )
            rewards = {"huntReward": huntReward, "onDefReward": onDefReward}

            self.updateWeights(self.prevState, self.prevAction, gameState, rewards)

            # store weights into JSON file
            if gameState.isOver():
                self.storeWeights()

        # get best action
        actions = gameState.getLegalActions(self.index)
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [
            self.evaluate(gameState, a, DEFENSE_WEIGHTS_ALL_FEATURES) for a in actions
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
        enemies = [currentState.getAgentState(i) for i in self.getOpponents(currentState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        prevEnemies = [prevState.getAgentState(i) for i in self.getOpponents(prevState)]
        prevInvaders = [a for a in prevEnemies if a.isPacman and a.getPosition() != None]
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            closestInvaderDistance = min(dists)
        if len(prevInvaders) > 0:
            prevDists = [self.getMazeDistance(prevPos, a.getPosition()) for a in prevInvaders]
            prevClosestInvaderDistance = min(prevDists)

        if len(invaders) > 0 and len(prevInvaders) > 0 and closestInvaderDistance < prevClosestInvaderDistance:
            reward = 0.01

        if len(invaders) > 0:
            if closestInvaderDistance == 0:
                reward = 0.1
        return reward
    
    def getDefendingReward(self, prevState, prev_action, currentState):
        reward = 0.001
        if currentState.getAgentState(self.index).isPacman:
            reward = 0
        return reward

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        prevPos = gameState.getAgentState(self.index).getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features["onDefense"] = 1
        if myState.isPacman:
            features["onDefense"] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        prevEnemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        prevInvaders = [a for a in prevEnemies if a.isPacman and a.getPosition() != None]
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            closestInvaderDistance = min(dists)
        if len(prevInvaders) > 0:
            prevDists = [self.getMazeDistance(prevPos, a.getPosition()) for a in prevInvaders]
            prevClosestInvaderDistance = min(prevDists)

        if len(invaders) > 0 and len(prevInvaders) > 0 and closestInvaderDistance < prevClosestInvaderDistance:
            features["gotCloserToInvader"] = 1.0
        else:
            features["gotCloserToInvader"] = 0.0

        if len(invaders) > 0:
            if closestInvaderDistance == 0:
                features["ateInvader"] = 1.0
        else:
            features["ateInvader"] = 0.0
        


        

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
            self.evaluate(nextState, a, DEFENSE_WEIGHTS_HUNT_FEATURES)
            for a in actions
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
            self.weights[feature] += (
                LEARNING_RATE * huntDifference * features[feature]
            )

        onDefDifference = (
            rewards["onDefReward"] + DISCOUNT_RATE * maxOnDefValue
        ) - self. evaluate(state, action, DEFENSE_WEIGHTS_STAYDEF_FEATURES)

        for feature in DEFENSE_WEIGHTS_STAYDEF_FEATURES:
            self.weights[feature] += LEARNING_RATE * onDefDifference * features[feature]
    
    #TODO TODO TODO change paths to new ones TODO TODO TODO
    def storeWeights(self):
        weights_json = json.dumps(self.weights, indent=4)
        with open(TRAINING_WEIGHT_PATH, "w") as outfile:
            outfile.write(weights_json)

    def loadWeights(self):
        try:
            with open(TRAINING_WEIGHT_PATH, "r") as file:
                self.weights = json.load(file)
        except IOError:
            print("Weights file not found.")

    def initializeWeights(self):
        self.weights = {}
        for key in DEFENSE_WEIGHTS_ALL_FEATURES:
            self.weights[key] = 0.0


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

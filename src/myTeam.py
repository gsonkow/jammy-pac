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
OFFENSE_WEIGHT_PATH = "weights_MY_TEAM.json"

# Any other constants used for your training (learning rate, discount, etc.)
# should be specified here
# [!] TODO
# TODO learning rate, discount, etc
# FRESH_START = False?

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
        #! Forney am I misreading this? Seems like it should be for offense?
        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            # UPDATE WEIGHTS
            if TRAINING:
                # TODO
                reward = self.getReward(gameState, bestAction)
                # TODO
                self.updateWeights(reward, self.getFeatures(gameState, bestAction))
            return bestAction

        # TODO - helperize this updatewieghts thing
        if TRAINING:
            reward = self.getReward(gameState, bestAction)
            self.updateWeights(reward, self.getFeatures(gameState, bestAction))

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
        return self.weights

    def loadWeights(self):
        try:
            with open(WEIGHT_PATH, "r") as file:
                self.weights = json.load(file)
        except IOError:
            print("Weights file not found, using default weights.")
            # TODO need to actually make this happen on error

    def initializeWeights(self):
        self.weights = {"successorScore": 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        if exists(OFFENSE_WEIGHT_PATH):
            self.load_weights()
        else:
            self.initialize_weights()

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        food = self.getFood(successor)
        foodList = food.asList()
        prev_food = self.getFood(gameState)
        prev_food_list = prev_food.asList()
        features["successorScore"] = -len(foodList)  # self.getScore(successor)
        myPos = successor.getAgentState(self.index).getPosition()

        # Compute distance to the nearest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minFoodDistance = min(
                [self.getMazeDistance(myPos, food) for food in foodList]
            )
            features["distanceToFood"] = minFoodDistance

        # ? Compute distance to the nearest food in prev state (should HELPER this and above, maybe not before vs for each action)
        if (
            len(prev_food_list) > 0
        ):  # This should always be True,  but better safe than sorry
            minPrevFoodDistance = min(
                [self.getMazeDistance(myPos, food) for food in prev_food_list]
            )

        if len(prev_food_list) > 0 and len(foodList) > 0:
            if minPrevFoodDistance < minFoodDistance:
                features["gotCloserToFood"] = 1.0
            else:
                features["gotCloserToFood"] = 0.0

        # encode if pellet eaten by action
        if prev_food.count() > food.count():
            features["pellet_eaten"] = 1.0
        else:
            features["pellet_eaten"] = 0.0
        # Better baselines will avoid defenders!
        enemies = [
            successor.getAgentState(opponent)
            for opponent in self.getOpponents(successor)
        ]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(ghosts) > 0:
            minGhostDistance = (
                min([self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]) + 1
            )
            features["distanceToGhost"] = minGhostDistance

        # Can't stop won't stop
        if action == Directions.STOP:
            features["stop"] = 1

        # Safety in numbers! ...even if that number is 2
        friends = [
            successor.getAgentState(friend) for friend in self.getTeam(successor)
        ]
        minFriendDistance = (
            min([self.getMazeDistance(myPos, a.getPosition()) for a in friends]) + 1
        )
        features["separationAnxiety"] = minFriendDistance

        # ! where was the logic to not eat pellet if ghost near? is that here?
        return features

    def getWeights(self, gameState, action):
        return {
            "successorScore": 0,
            "distanceToFood": 0,
            "gotCloserToFood": 0,
            "pellet_eaten" "distanceToGhost": 0,
            "stop": 0,
            "separationAnxiety": 0,
        }

    def loadWeights(self):
        try:
            with open(OFFENSE_WEIGHT_PATH, "r") as file:
                self.weights = json.load(file)
        # TODO need to actually make this happen on error

        except IOError:
            print("Weights file not found, using default weights.")

    def initializeWeights(self):
        self.weights = {
            "successorScore": 0,
            "distanceToFood": 0,
            "gotCloserToFood": 0,
            "pellet_eaten" "distanceToGhost": 0,
            "stop": 0,
            "separationAnxiety": 0,
        }


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        if exists(WEIGHT_PATH):
            self.load_weights()
        else:
            self.initialize_weights()

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

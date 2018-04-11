# Simulation framework for two blockchains
import random

## 2^256 / F_max
max_ratio = 4.29 * 10**9

# A Period contains all of the information to specify a period of time on a
# chain during which distinct rates of profit exist for the switching miner.
# This includes:
#     1. Hashrate mining during this period
#     2. Block difficulty
#     3. Blocks mined during the period
#     4. Time it took for the chain to complete this period
class Period:
    def __init__(self, hashRate, diff, time, blocks):
        self.hashRate = hashRate
        self.diff = diff
        self.time = time
        self.blocks = blocks
        return

# A Chain contains all of the information needed to specify a blockchain for
# the simulation.  This includes:
#     1. Global parameters (loyal hashrate, switching rate, target time)
#     2. The periods in the chain's history
#     3. A function to draw the next time it takes to mine a difficulty epoch
class Chain:
    def __init__(self, alpha, beta, targetTime, blockNum, hashRate, tokenValue):
        self.alpha = alpha
        self.beta = beta
        self.targetTime = targetTime
        self.blockNum = blockNum
        self.hashRate = hashRate
        self.tokenValue = tokenValue
        self.diffPeriods = 0
        self.diff = targetTime * 1.0 * (1 / max_ratio) * hashRate
        self.periods = [Period(hashRate, self.diff, targetTime, blockNum)]
        return

    # Update the state of the chain to account for the specified period of
    # blocks being mined
    def periodMined(self, blocks, time):
        self.periods.append(Period(self.hashRate, self.diff, time, blocks))
        return

    # Update the current difficulty of the chain
    def updateDifficulty(self, time):
        self.diffPeriods += 1
        self.diff = (self.diff * self.targetTime * self.blockNum) / time
        return

    # Update the state of the chain to account for a new allocation of hash
    # power
    def updateHash(self, newRate):
        self.hashRate = newRate

    # nextTime calculates the time until the next difficulty adjustment
    # period.
    #
    # In the case the previous period ended on the boundary of a difficulty
    # adjustment period it draws the time to completion of the next period from
    # from exponential distribution with mean around XXX
    #
    # In the case the previous period did not finish out an entire difficulty
    # adjustment period the mean is around YYY
    def nextTime(self):
        prev_diff = self.periods[-1].diff
        mean = (max_ratio * prev_diff) / self.hashRate
        return rand.expovariate(1.0 / mean)

# The greedy switching strategy decision function
def gDecision(chain1, chain2):
    alpha = chain1.alpha
    # The time to complete this difficulty adj period if miner switches to this
    #  chain
    prev_diff1 = chain1.periods[-1].diff
    prev_diff2 = chain2.periods[-1].diff
    expectedTime1 = (max_ratio * prev_diff1) / chain1.hashRate
    expectedTime2 = (max_ratio * prev_diff2) / chain2.hashRate

    rate1 = (chain1.tokenValue / expectedTime1) * (alpha / (chain1.hashRate))
    rate2 = (chain2.tokenValue / expectedTime2) * (alpha / (chain2.hashRate))

    if rate1 >= rate2:
        return alpha, 0
    else:
        return 0, alpha

# A Simulation carries the information required to run a simulation with a
# given configuration.  This includes:
#     1. Parameter bounds
#     2. Step sizes
#     3. Output configuration
#     4. Number of difficulty adjustment periods to simulate
class Simulation:
    def __init__(self, numAdjPeriods):
        self.numAdjPerods = numAdjPeriods
        random.seed()
        return
    
    def runGreedy(self):
        # TODO sweep in alpha

        # TODO sweep in betaFrac

        # TODO sweep in priceFrac
        return

    # runOne executes mining on two chains for a number of difficulty 
    # adjustment periods, returning the two Chains and associated data for
    # processing.  It is parameterized by a decision function and can
    # therefore be used to simulate different switching-miner strategies
    def runOne(self, beta1, beta2, alpha, f1, f2, decisionFun):
        # We constrain the problem a bit to simplify things:
        #  alpha always starts out on chain 1
        chain1 = Chain(beta1, alpha, 600, 2016, beta1 + alpha, f1)
        chain2 = Chain(beta2, alpha, 600, 2016, beta2, f2)
        # Each loop run adds a period to each chain
        # Difficulty adjustments and periods are not 1-1 so i is not updated
        #  in every loop
        while chain1.diffPeriods < self.numAdjPeriods and
        chain2.diffPeriods < self.numAdjPeriods:
            time1 = chain1.nextTime()
            time2 = chain2.nextTime()
            minTime = min(time1, time2)

            # Update the periods
            blocks1 = math.floor((minTime / time1) * chain1.blockNum)
            blocks2 = math.floor((minTime / time2) * chain2.blockNum)
            chain1.periodMined(blocks1, minTime)
            chain2.periodMined(blocks2, minTime)

            # Difficulty of chain 1 adjusts
            if time1 == minTime:
                chain1.updateDifficulty(time1)
                
            # Difficulty of chain 2 adjusts
            if time2 == minTime:
                chain2.updateDifficulty(time2)

            # Switching miner makes her decision
            alpha1, alpha2 = decisionFun(chain1, chain2)

            chain1.updateHash(alpha1 + beta1)
            chain2.updateHash(alpha2 + beta2)

        return chain1, chain2

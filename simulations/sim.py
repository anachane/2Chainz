# Simulation framework for two blockchains
import random
import math
import csv
import datetime
import matplotlib.pyplot as plt
import numpy as np

## 2^256 / F_max
max_ratio = 4.29 * 10**9
## arbitrarily one billion to make difficulties easier to talk about
hPerSecond = 1000000000

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
        self.timeSinceAdj = time
        self.blocks = blocks
        return

# A Chain contains all of the information needed to specify a blockchain for
# the simulation.  This includes:
#     1. Global parameters (loyal hashrate, switching rate, target time)
#     2. The periods in the chain's history
#     3. A function to draw the next time it takes to mine a difficulty epoch
class Chain:
    def __init__(self, alpha, beta, targetTime, blockNum, hashRate, tokenValue):
        # system parameters
        self.alpha = alpha
        self.beta = beta
        self.blockNum = blockNum
        self.tokenValue = tokenValue

        # current state of chain
        self.timeSinceAdj = 0.0
        self.timeThisPeriod = 0.0
        self.lastPuzzleTime = 0.0
        self.blocks = 0
        self.hashRate = hashRate
        self.targetTime = targetTime
        self.diff = targetTime * hPerSecond * (1 / max_ratio) * hashRate # init assuming steady state

        # history of chain
        self.diffPeriods = 0
        self.periods = [Period(hashRate, self.diff, targetTime, blockNum)]
        return

    # Update the state of the chain to account for the specified period of
    # blocks being mined
    def periodMined(self):
        self.periods.append(Period(self.hashRate, self.diff, self.timeThisPeriod, self.blocks))
        self.lastPuzzleTime = 0.0
        return

    # Update the current difficulty of the chain
    def updateDifficulty(self):
        self.diffPeriods += 1
        self.diff = (self.diff * self.targetTime * self.blockNum) / self.timeSinceAdj
        return

    # Update the state of the chain to account for a new allocation of hash
    # power
    def updateHashRate(self, newRate):
        self.hashRate = newRate

    # Chain mines a block with its current hashrate.  Updates the blocks mined
    # by this chain and the time spend by this chain on this period
    def mineBlock(self):
        prev_diff = self.periods[-1].diff
        # max_ratio * prev_diff == mean # of hashes until block is mined
        # mean time == mean # hashes / hash rate
        mean = (max_ratio * prev_diff) / (hPerSecond * self.hashRate)
        self.lastPuzzleTime = random.expovariate(1.0 / mean)
        self.blocks += 1
        self.timeThisPeriod += self.lastPuzzleTime
        self.timeSinceAdj += self.lastPuzzleTime

### Decision functions define a switching miner's strategy.  These functions
### take in the state of both chains at any difficulty adjustment in the system
### and return a hashpower allocation to both chains 

# The greedy switching strategy decision function
def gDecision(chain1, chain2):
    alpha = chain1.alpha
    prev_diff1 = chain1.periods[-1].diff
    prev_diff2 = chain2.periods[-1].diff

    rate1 = chain1.tokenValue / prev_diff1
    rate2 = chain2.tokenValue / prev_diff2

    if rate1 >= rate2:
        return alpha, 0
    else:
        return 0, alpha

### Data functions help process raw chain history data into measurements. Map
### functions take in the history of two chains run to completion over a given
### number of diff adjustment periods and return a number measuring the run in
### some way. Reduce functions take measurements of runs over the same 
### parameter values and combine them in some way, i.e. mean, max, stddev

### Map Funs
# countOscData counts the number of switches occurring in the trial
def countOscMap(ch1, ch2):
    oscs = 0
    lastP = None
    for period in ch1.periods:
        if lastP == None:
            lastP = period
            continue
        if lastP.hashRate != period.hashRate:
            oscs += 1
        lastP = period
    return oscs

### Reduce Funs
# mean takes the mean of data, either ints or floats.  Output float
def meanReduce(results):
    total = 0.0
    for data in results:
        total += data
    return total / float(len(results))

# max takes the max of data
def maxReduce(results):
    return max(results)

# A Simulation carries the information required to run a simulation with a
# given configuration.  This includes:
#     1. Parameter bounds
#     2. Step sizes
#     3. Output configuration
#     4. Number of difficulty adjustment periods to simulate
class Simulation:
    def __init__(self, numAdjPeriods):
        self.numAdjPeriods = numAdjPeriods
        random.seed()
        self.stepSize = 0.01
        self.runsPerStep = 5

        return

    # runGreedy runs a simulation with the greedy-switching miner strategy and
    # a provided dataFun that processes the results of the 
    def runGreedy(self, mapFun, reduceFun):
        alphaMap = {}
        for alpha in [0.01, 0.05, 0.1, 0.3]:
            beta1Map = {}
            alphaMap[alpha] = beta1Map
            beta1 = 0
            while beta1 < 1 - alpha:
                beta1 = beta1 + self.stepSize
                print("beta1: " +str(beta1))
                priceMap = {}
                beta1Map[beta1] = priceMap
                beta2 = (1 - alpha) - beta1
                priceFrac = 0 # f2/f1
                f2 = 1 # setting this for simulation simplicity
                while int(priceFrac) < 1:
                    priceFrac = priceFrac + self.stepSize
                    print("priceFrac: " + str(priceFrac))
                    results = []
                    f1 = f2 / priceFrac
                    for i in range(self.runsPerStep):
                        ch1, ch2 = self.runOne(beta1, beta2, alpha, f1, f2, gDecision)
                        results.append(mapFun(ch1, ch2))
                    priceMap[priceFrac] = reduceFun(results)
        return alphaMap

    # runOne executes mining on two chains for a number of difficulty 
    # adjustment periods, returning the two Chains and associated data for
    # processing.  It is parameterized by a decision function and can
    # therefore be used to simulate different switching-miner strategies
    def runOne(self, beta1, beta2, alpha, f1, f2, decisionFun):
        # We constrain the problem a bit to simplify things:
        #  alpha always starts out on chain 1
        chain1 = Chain(alpha, beta1, 600, 2016, beta1 + alpha, f1)
        chain2 = Chain(alpha, beta2, 600, 2016, beta2, f2)
        # Each loop run adds a period to each chain
        # Difficulty adjustments and periods are not 1-1 so i is not updated
        #  in every loop
        
        while chain1.diffPeriods < self.numAdjPeriods and chain2.diffPeriods < self.numAdjPeriods:
            # Mine until one difficulty must be adjusted
            
            while chain1.blocks < chain1.blockNum and chain2.blocks < chain2.blockNum:
                maxTime = max(chain1.timeThisPeriod, chain2.timeThisPeriod)
                if chain1.timeThisPeriod < maxTime:
                    chain1.mineBlock()
                else:
                    chain2.mineBlock()
            if chain1.blocks == chain1.blockNum:
                self.finishMiningPeriod(chain1, chain2)
            else:
                self.finishMiningPeriod(chain2, chain1)
                
            # Update the periods
            chain1.periodMined()
            chain2.periodMined()
            
            # Difficulty of chain 1 adjusts and reset adj values
            if chain1.blocks == chain1.blockNum:
                chain1.updateDifficulty()
                chain1.blocks = 0
                chain1.timeSinceAdj = 0.0

            # Difficulty of chain 2 adjusts
            if chain2.blocks == chain2.blockNum:
                chain2.updateDifficulty()
                chain2.blocks = 0
                chain2.timeSinceAdj = 0.0

            chain1.timeThisPeriod = 0.0
            chain2.timeThisPeriod = 0.0

            # Switching miner makes her decision
            alpha1, alpha2 = decisionFun(chain1, chain2)
            
            chain1.updateHashRate(alpha1 + beta1)
            chain2.updateHashRate(alpha2 + beta2)

        return chain1, chain2

    # Helper function to address edge cases at the end of mining periods.
    # The non-difficulty-adjusting chain may not be finished as the other chain
    # mined last. Furthermore the non-difficulty-adjusting chain may acutally
    # finish first.  A block mined beyond the period boundary is erased
    def finishMiningPeriod(self, chainA, chainB):
        chainBMines = False
        while chainB.timeThisPeriod < chainA.timeThisPeriod and chainB.blocks < chainB.blockNum:
            chainBMines = True            
            chainB.mineBlock()
        if chainBMines and chainB.timeThisPeriod > chainA.timeThisPeriod:
            chainB.blocks -= 1
            chainB.timeThisPeriod -= chainB.lastPuzzleTime
            chainB.timeSinceAdj -= chainB.lastPuzzleTime
        if chainB.timeThisPeriod < chainA.timeThisPeriod and chainB.blocks == chainB.blockNum:
            if not chainBMines: # chainB could not have won the race originally if chainA won the race originally
                print("Should not get here")
                assert(false)
            chainA.blocks -= 1
            chainA.timeThisPeriod -= chainA.lastPuzzleTime
            chainA.timeSinceAdj -= chainA.lastPuzzleTime

### Output functions save and visualize data from a simulation run
    # plotResults plots the simulation measurements in a matplotlib graph.
    # Plots are formatted to display beta vs priceFrac graphs for each alpha value
    # using the hexbin matplotlibe utility to tile the surface
    def plotResults(self, resultMap):
        for alpha in resultMap:
            # set up matplotlib graph
            ncols = int((1-alpha)/self.stepSize)
            nrows = int(1.1/self.stepSize)
            vals = np.zeros((nrows,ncols))
            betaMap = resultMap[alpha]
            for ib,beta1 in enumerate(sorted(betaMap)):
                if ib > ncols -1:
                    print("ib: " + str(ib) + "beta1: " + str(beta1))
                    break
                priceMap = betaMap[beta1]
                for ip,priceFrac in enumerate(sorted(priceMap)):
                    if ip > nrows -1:
                        print("ip: " + str(ip) + "priceFrac: " + str(priceFrac))
                        break
                    val = priceMap[priceFrac]
                    vals[ip][ib] = val
            plt.pcolor(vals)
            plt.show()
            plt.savefig("alpha="+str(alpha)+".png")

                     
# saveResults saves the simulation measurements to a csv file in the form
# alpha, beta1, priceFrac, dataValue
def saveResults(resultMap):
    now = datetime.datetime.now()
    fileName = str(now)[:16]
    with open(fileName, 'w') as csvfile:
        outputWriter = csv.writer(csvfile, delimiter=',')
        for alpha in resultMap:
            betaMap = resultMap[alpha]
            for beta1 in betaMap:
                priceMap = betaMap[beta1]
                for priceFrac in priceMap:
                    dataValue = priceMap[priceFrac]
                    outputWriter.writerow([str(alpha), str(beta1),
                                           str(priceFrac), str(dataValue)])

# loadResults loads a csv file to a python results dictionary                    


# Run from cli
if __name__ == "__main__":
    sim = Simulation(100)
    resultMap = sim.runGreedy(countOscMap, meanReduce)
    saveResults(resultMap)
    sim.plotResults(resultMap)

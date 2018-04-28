import csv
import sim
import numpy as np
import matplotlib.pyplot as plt

class Analysis:
    # take in the map, reduce function names, the stepsize between data and dataMap
    def __init__(self, stepSize, data, titleStr):
       self.stepSize = stepSize
       self.resultMap = data
       self.titleStr = titleStr

    # Output functions save and visualize data from a simulation run
    # plotResults plots the simulation measurements in a matplotlib graph.
    # Plots are formatted to display beta vs priceFrac graphs for each alpha value
    # using the hexbin matplotlibe utility to tile the surface
    def plotResults(self):
        for alpha in self.resultMap:
            # set up matplotlib graph
            ncols = int((1-alpha)/self.stepSize)
            nrows = int(1.1/self.stepSize)
            vals = np.zeros((nrows,ncols))
            betaMap = self.resultMap[alpha]
            for ib,beta1 in enumerate(sorted(betaMap)):
                if ib > ncols -1:
                    print("ib: " + str(ib) + "beta1: " + str(beta1))
                    continue
                priceMap = betaMap[beta1]
                for ip,priceFrac in enumerate(sorted(priceMap)):
                    if ip > nrows -1:
                        print("ip: " + str(ip) + "priceFrac: " + str(priceFrac))
                        continue
                    val = priceMap[priceFrac]
                    vals[ip][ib] = val
            fig, axis = plt.subplots()
            axis.set_xlabel("Beta1")
            axis.set_ylabel("f2/f1")
            axis.set_title(self.titleStr + "-alpha=" + str(alpha))
            heatmap = axis.pcolor(vals)
            plt.colorbar(heatmap)
            plt.savefig(self.titleStr +"-alpha="+str(alpha)+".png", dpi=100)
            plt.show()
            
# loadResults loads a csv file to a python results dictionary
def loadResults(fileName, mapFun, reduceFun):
    data = {}
    resultsMap = {}    
    with open(fileName, 'r') as csvfile:
        resultsReader = csv.reader(csvfile)
        # iterate through lines        
        for i, row in enumerate(resultsReader):
            print("processing row: " + str(i))
            # values for alpha beta and priceFrac
            alpha = float(row[0])
            beta1 = float(row[1])
            priceFrac = float(row[2])
            if row[3] != "trial0" or row[4] != "chain0":
                print("unexpected 4th element: " + str(row[3]))
                print("or unexpected 5th element: " + str(row[4]))                
                return
            idx = 4
            vals = []
            # iterate over trials until no data is left
            while True:
                ch1 = sim.Chain(0, 0, 0, 0, 0, 0) # only periods matter
                ch1.periods = [] #erase first period made on startup
                ch2 = sim.Chain(0, 0, 0, 0, 0, 0) # only periods matter
                ch2.periods = [] #erase first period made on startup
                while idx < len(row) and "trial" not in str(row[idx]):
                    if "chain" in str(row[idx]) and "trial" in str(row[idx-1]):
                        chain = ch1
                        idx += 1
                        continue
                    elif "chain" in str(row[idx]):
                        chain = ch2
                        idx += 1
                        continue
                    chain.periods.append(sim.Period(row[idx], row[idx+1], row[idx+2], row[idx+3]))
                    idx += 4
                # process completed chains
                ch1.alpha = alpha
                ch1.beta = beta1
                ch1.tokenValue = 1.0/priceFrac
                ch2.alpha = alpha
                ch2.beta = 1 - alpha - beta1
                ch2.tokenValue = 1.0
                vals.append( mapFun(ch1, ch2) )
                if idx == len(row):
                    break
                idx += 1

            # combine data and add to output datastructure
            val = reduceFun(vals)
            if alpha not in resultsMap:
                resultsMap[alpha] = {}
            alphaMap = resultsMap[alpha]

            if beta1 not in alphaMap:
                alphaMap[beta1] = {}
            betaMap = alphaMap[beta1]

            betaMap[priceFrac] = val
        
    return resultsMap
            

### Map Funs
# countOsc counts the number of switches occurring in the trial
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

# countOsc 1s and 0s assigns three values, one value for 0 oscillations
# on value for 1 oscillation and a third value for more than this
def countOsc10Map(ch1, ch2):
    oscs = countOscMap(ch1, ch2)
    if oscs == 0:
        return 1
    if oscs == 1:
        return 2
    else:
        return 0

# countProfit calculates the profit of the switching miner over the history
# of both chains.  Normalized by expected profit of mining on chain1 without
# switching for the given time
def countProfitMap(ch1, ch2):
    expectedProfit = 0.0
    tNorm = finishTimeMap(ch1, ch2)
    profitNorm = (tNorm / 600.0) * (ch1.alpha / ch1.beta + ch1.alpha) * ch1.tokenValue
                       
    for period in ch1.periods:
        switchHash = float(period.hashRate) - ch1.beta
        if switchHash > 0.0:
            expectedProfit += ch1.tokenValue * int(period.blocks) * (switchHash / float(period.hashRate))

    for period in ch2.periods:
        switchHash = float(period.hashRate) - ch2.beta
        if switchHash > 0.0:
            expectedProfit += ch2.tokenValue * int(period.blocks) * (switchHash / float(period.hashRate))
    return expectedProfit / profitNorm

# diffLeap returns 1 if there is a difficulty adjustment greater than 4x in
# either direction.  If this happens a lot then that means simulations should
# probably be rerun.  This will always be 0 with new difficulty adjustment
# calculation fixes
def diffLeapMap(ch1, ch2):
    lastDiff = None
    for period in ch1.periods:
        if lastDiff is None:
            lastDiff = float(period.diff)
            continue
        if float(period.diff) < lastDiff / 4.0 or float(period.diff) > lastDiff * 4.0:
            return 1
        lastDiff = float(period.diff)

    lastDiff = None
    for period in ch2.periods:
        if lastDiff is None:
            lastDiff = float(period.diff)
            continue
        if float(period.diff) < lastDiff / 4.0 or float(period.diff) > lastDiff * 4.0:
            return 1
        lastDiff = float(period.diff)

    return 0

# finishTime returns the simulated time it took for one of the chains to
# complete 100 difficulty adjustments.  Note this function needs to change
# when analyzing runs of greater or fewer than 100 diff adjustments
def finishTimeMap(ch1, ch2):
    totalTime1 = 0.0
    for period in ch1.periods:
        totalTime1 += float(period.timeSinceAdj) # TODO fix misnomer in sim code this is actually just time in period

    totalTime2 = 0.0
    for period in ch2.periods:
        totalTime2 += float(period.timeSinceAdj)
        
    return min(totalTime1, totalTime2)

# blockDiff returns the number of blocks mined by ch2 minus the number of
# blocks mined by ch1
def blockDiffMap(ch1, ch2):
    blocks1 = 0
    for period in ch1.periods:
        blocks1 += int(period.blocks)
    blocks2 = 0
    for period in ch2.periods:
        blocks2 += int(period.blocks)
    return blocks2 - blocks1

# avgHahsPowerCh2 returns the ratio of the average hash rate on chain2 over
# its history and the max possible hash rate (alpha + beta2)
def avgHashPowerCh2Map(ch1, ch2):
    avgHashRate = float(ch2.periods[0].hashRate)
    for period in ch1.periods[1:]:
        avgHashRate += float(period.hashRate)

    avgHashRate /= len(ch2.periods)
    normalized = avgHashRate / (ch2.alpha + ch2.beta)
    return normalized
    

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

def extreme01Reduce(results):
    mean = meanReduce(results)
    if mean != 0 and mean != 1 and mean != 2:
        return 0
    return mean

def stdReduce(results):
    return np.std(results)

            
if __name__ == "__main__":
   data = loadResults("results/big-run-0", avgHashPowerCh2Map, meanReduce)
   analysis = Analysis(0.01, data, "normalized-hashrate")
   analysis.plotResults()

import csv
import sim
import numpy as np
import matplotlib.pyplot as plt

class Analysis:
    # take in the map, reduce function names, the stepsize between data and dataMap
    def __init__(self, stepSize, data):
       self.stepSize = stepSize
       self.resultMap = data

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
            axis.set_title("mean # oscillations, alpha=" + str(alpha))
            heatmap = axis.pcolor(vals)
            plt.colorbar(heatmap)
            plt.savefig("alpha="+str(alpha)+".png", dpi=100)
            plt.show()
            
# loadResults loads a csv file to a python results dictionary
def loadResults(fileName, mapFun, reduceFun):
    data = {}
    resultsMap = {}    
    with open(fileName, 'r') as csvfile:
        resultsReader = csv.reader(csvfile)
        # iterate through lines        
        for i, row in enumerate(resultsReader):
            print("row: " + str(i) + " has len: " + str(len(row) ))
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
            
if __name__ == "__main__":
   data = loadResults("results/big-run-0", countOscMap, meanReduce)
   analysis = Analysis(0.01, data)
   analysis.plotResults()

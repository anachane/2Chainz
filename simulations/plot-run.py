import sim
import matplotlib.pyplot as plt
import csv

def plotDataSets(x1, y1, x2, y2, xlabel, ylabel, title, fileName):
    fig, axis = plt.subplots()
    axis.plot(x1, y1, c='b', label='chain1')
    axis.plot(x2, y2, c='r', label='chain2')
    plt.legend(loc='upper left')
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)

    axis.set_title(title)
    plt.savefig(fileName+".png", dpi=100)
    plt.show()
    
# Plot one point at the end of each period.  X axis measures number of blocks
# mined to get to that point, Y axis measures hashrate on chain during that
# period
def plotHashBlocks(chain1, chain2):
    blockNum1 = []
    hashRates1 = []
    blockTotal1 = 0
    blockNum2 = []
    hashRates2 = []
    blockTotal2 = 0    
    for period in chain1.periods:
        blockTotal1 += period.blocks
        blockNum1.append(blockTotal1)
        hashRates1.append(period.hashRate)

    for period in chain2.periods:
        blockTotal2 += period.blocks
        blockNum2.append(blockTotal2)
        hashRates2.append(period.hashRate)

    fn = "single-run-results/hashrate-over-blocks" + "-alpha=" + str(chain1.alpha) + "-beta1=" + str(chain1.beta) + "-token1Value=" + str(chain1.tokenValue)
    plotDataSets(blockNum1, hashRates1, blockNum2, hashRates2, "Blocks mined", "Hash rate", "Hashrate over Blocks", fn)

# Plot one point at the end of each period.  X axis measures number of blocks
# mined to get to that point, Y axis measures Difficulty on chain during that
# period
def plotDiffBlocks(chain1, chain2):
    blockNum1 = []
    Diffs1 = []
    blockTotal1 = 0
    blockNum2 = []
    Diffs2 = []
    blockTotal2 = 0    
    for period in chain1.periods:
        blockTotal1 += period.blocks
        blockNum1.append(blockTotal1)
        Diffs1.append(period.diff)

    for period in chain2.periods:
        blockTotal2 += period.blocks
        blockNum2.append(blockTotal2)
        Diffs2.append(period.diff)

    fn = "single-run-results/diff-over-blocks" + "-alpha=" + str(chain1.alpha) + "-beta1=" + str(chain1.beta) + "-token1Value=" + str(chain1.tokenValue)
    plotDataSets(blockNum1, Diffs1, blockNum2, Diffs2, "Blocks mined", "Difficulty", "Difficulty over Blocks", fn)        


# loadResults loads a csv file to a python results dictionary
def traverseData(fileName, params):
    with open(fileName, 'r') as csvfile:
        resultsReader = csv.reader(csvfile)
        # iterate through lines        
        for i, row in enumerate(resultsReader):
            print("processing row: " + str(i))
            # values for alpha beta and priceFrac
            alpha = float(row[0])
            beta1 = float(row[1])
            priceFrac = float(row[2])
            if abs(alpha - params[0]) > 0.001 or abs(beta1 - params[1]) > 0.001 or abs(priceFrac - params[2]) > 0.001:
                continue
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
                    chain.periods.append(sim.Period(float(row[idx]), float(row[idx+1]), float(row[idx+2]), int(row[idx+3])))
                    idx += 4
                # process completed chains
                ch1.alpha = alpha
                ch1.beta = beta1
                ch1.tokenValue = 1.0/priceFrac
                ch2.alpha = alpha
                ch2.beta = 1 - alpha - beta1
                ch2.tokenValue = 1.0

                # plot hashpower and difficulty over blocks
                plotHashBlocks(ch1, ch2)
                plotDiffBlocks(ch1, ch2)
                return


if __name__ == "__main__":
    filename = "results/big-run-1"
    # (alpha, beta, priceFrac) to plot 
    params = (0.2, 0.5, 0.7)
    traverseData(filename, params)

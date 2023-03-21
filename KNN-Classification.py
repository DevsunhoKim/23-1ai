import csv
import math
import operator
import random
import multiprocessing
import time
import sklearn.model_selection
import re


with open('housing.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        print(row)


class KNNCalculator:
    lock = multiprocessing.Lock()

    def __init__(self, dataSet, comparator, kLimit=1, thread=1, skipFirst=False, paralel=False, id=0, sharedList=None):
        """dataSet can be string to indicate file or a list to indicate data, comparator must be supplied to compute similarity"""

        self.comparator = comparator
        self.thread = thread
        self.kLimit = kLimit
        self.paralel = paralel
        self.skipFirst = skipFirst
        self.testSet = []
        self.trainSet = []
        self.id = id
        self.sharedList = sharedList

        if type(dataSet) is str:
            file = open(dataSet)
            if skipFirst:
                self.header = str(file.readline()).split(sep=',')
            self.data = list(csv.reader(file))
            for i in range(len(self.data)):
                for j in range(len(self.data[i])-1):
                    self.data[i][j] = float(self.data[i][j])  
        elif type(dataSet) is list:
            self.data = dataSet

    def RandomizeDataSet(self, ratio):
        for dataItem in self.data:
            if random.random() < ratio:
                self.testSet.append(dataItem)
            else:
                self.trainSet.append(dataItem)

    def GetNeighbors(self, instance):
        distance = list()
        neighbors = list()

        for i in range(len(self.trainSet)):
            res = self.comparator(instance, self.trainSet[i])
            distance.append((self.trainSet[i], res))
        distance.sort(key=operator.itemgetter(1))
        for i in range(self.kLimit):
            neighbors.append(distance[i][0])
        return neighbors

    def GetResponse(self, neighbors):
        classVotes = {}
        for data in neighbors:
            if data[-1] in classVotes:
                classVotes[data[-1]] += 1
            else:
                classVotes[data[-1]] = 1
        res = sorted(classVotes.items(),
                     key=operator.itemgetter(1), reverse=True)
        return res[0][0]

    def GetAccuracy(self, prediction):
        correct = 0
        for i in range(len(self.testSet)):
            if self.testSet[i][-1] == prediction[i]:
                correct += 1
        return (correct/float(len(self.testSet))) * 100.0

    @staticmethod
    def MinMaxNormalize(data):
        for i in range(len(data[0])-1):
            minData = float(min(data, key=operator.itemgetter(i))[i]) 
            maxData = float(max(data, key=operator.itemgetter(i))[i]) 
            for j in range(len(data)):
                data[j][i] = (float(data[j][i]) - minData)/(maxData-minData)

        return data

    @staticmethod
    def Split(iData, count=2):
        """Return (Z)(x,y) where Z is split Index X test data and Y train data"""
        dataEachSplit = int(len(iData)/count)
        res = list()
        res.append(list())
        
        for i in range(count - 1):
            res.append(list())
            testSet = iData[i*dataEachSplit:(i+1)*dataEachSplit]
            
            if i == 0:
                trainSet = iData[(i+1)*dataEachSplit:len(iData)]
            else:
                trainSet = iData[0:i*dataEachSplit]
                
            if i > 0:
                trainSet.extend(iData[(i+1)*dataEachSplit:len(iData)])
                
            res[i].append(testSet)
            res[i].append(trainSet)
            
        testSet = iData[(count-1)*dataEachSplit:len(iData)]
        trainSet = iData[0:(count-1)*dataEachSplit]
        
        res[count-1].append(testSet)
        res[count-1].append(trainSet)
        
        return res

    @staticmethod
    def StratifiedSplit(iData, count=2):
        skf = sklearn.model_selection.StratifiedKFold(count)
        outcome = list()
        
        for data in iData:
            outcome.append(data[-1])
            
        skf.get_n_splits(iData, outcome)
        res = list()
        i = 0
        
        for trainIndex, testIndex in skf.split(iData, outcome):
            res.append(list())
            res[i].append(list())
            res[i].append(list())
            
            for index in testIndex:
                res[i][0].append(iData[index])
                
            for index in trainIndex:
                res[i][1].append(iData[index])
                
            i += 1
            
        return res

    def Compute(self, splitMethod=None):
        if self.id == 0 and self.thread != 1:
            if splitMethod is None:
                raise("Split Method Must be supplied")
            
            threadPool = list()
            manager = multiprocessing.Manager()
            self.sharedList = manager.list()
            dataSplitted = splitMethod(self.data, self.thread)
            
            for i in range(self.thread):
                self.sharedList.append(0)
                currentKNN = KNNCalculator(
                    self.data, self.comparator, self.kLimit, 1, self.skipFirst, self.paralel, i, self.sharedList)
                currentKNN.testSet = dataSplitted[i][0]
                currentKNN.trainSet = dataSplitted[i][1]
                
                if not self.paralel:
                    currentKNN.Compute()
                else:
                    workerThread = multiprocessing.Process(target=currentKNN.Compute)
                    workerThread.start()
                    threadPool.append(workerThread)
                    
            self.thread = 1
            if self.paralel:
                for threads in threadPool:
                    threads.join()
                    
            res = 0
            listRes = list()
            for acc in self.sharedList:
                res += acc
                listRes.append(acc)
                
            finalRes = res/len(self.sharedList)
            return (finalRes, listRes)
        
        else:
            if self.sharedList is None:
                self.RandomizeDataSet(0.66)
                
            prediction = []
            for dataItem in self.testSet:
                neighbors = self.GetNeighbors(dataItem)
                res = self.GetResponse(neighbors)
                prediction.append(res)
                
            res = self.GetAccuracy(prediction)
            if self.sharedList is None:
                return (res, [res])
            
            self.sharedList[self.id] = res


def cosineSimilarity(lhs, rhs):
    if len(lhs) != len(rhs):
        raise Exception("Length of LHS and RHS are not the same!")
    numerator = 0
    denominator = [0, 0]
    for i in range(len(lhs)-1):
        numerator += lhs[i] * rhs[i]
        denominator[0] += lhs[i] * lhs[i]
        denominator[1] += rhs[i] * rhs[i]
    return 1-(numerator/(math.sqrt(denominator[0]) * math.sqrt(denominator[1])))


def euclideanDistance(lhs, rhs):
    distance = 0
    for i in range(len(lhs)-1):
        distance += pow(lhs[i]-rhs[i], 2)
    return math.sqrt(distance)


def manhattanDistance(lhs, rhs):
    distance = 0
    for i in range(len(lhs)-1):
        distance += abs(lhs[i]-rhs[i])
    return distance


def minkowskiDistance(lhs, rhs):
    distance = 0
    for i in range(len(lhs)-1):
        distance += abs(pow(lhs[i]-rhs[i], 3))
    return math.pow(distance, 1./3)


if __name__ == '__main__':
    fileName = input("File Name : ")
    kLimit = int(input("K-NN Value : "))
    threadCount = int(input("Cross Validator Count : "))

    KNNInstance = KNNCalculator(fileName, None, skipFirst=True)
    dataset = KNNCalculator.MinMaxNormalize(KNNInstance.data)

    distanceMethod = [euclideanDistance, manhattanDistance,
                      minkowskiDistance, cosineSimilarity]
    for method in distanceMethod:
        for kVal in range(kLimit):
            print("Computing Using " + method.__name__ + " K = " + str(kVal+1), end=" ")
            KNNNode = KNNCalculator(
                dataset, comparator=method, kLimit=kVal+1, thread=threadCount, id=0, paralel=True)
            res = KNNNode.Compute(splitMethod=KNNCalculator.StratifiedSplit)
            # for data in res[1]:
            #print("Accuracy : %.2f" % data)
            print("Average For %s K = %d: %.2f" %(method.__name__, kVal+1, res[0]))
            
# 해당 도서 코드 미리 받아보기 실습 하는법 질문하기
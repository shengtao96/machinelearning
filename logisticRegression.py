from numpy import *

def loadDataSet(filename):
	fr = open(filename)
	numFeat = len(fr.readline().strip().split(',')) - 1
	dataSet = []; labelSet = []
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split(',')
		lineArr.append(1.0)
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		dataSet.append(lineArr)
		labelSet.append(float(curLine[-1]))
	return dataSet, labelSet

def featureNormalize(dataMat):
	meanData = mean(dataMat, 0)
	meanData[0, 0] = 0.0
	stdData = std(dataMat, 0)
	stdData[0, 0] = 1.0
	normDataMat = (dataMat - meanData) / stdData
	return normDataMat, meanData, stdData

def sigmoid(dataSet):
	return 1/ (1 + exp(-dataSet))

def gradientDescent(dataMat, labelMat, alpha, num_iters = 50000):
	numFeat = shape(dataMat)[1]
	m = shape(dataMat)[0]
	theta = mat(zeros((numFeat, 1)))
	for i in range(num_iters):
		tmp = sigmoid(dataMat * theta) - labelMat
		theta -=  dataMat.T * tmp * alpha / m
	return theta

dataSet, labelSet = loadDataSet('D:\\machine-learning-ex2\\ex2\\ex2data1.txt')
dataMat = mat(dataSet)
labelMat = mat(labelSet)
labelMat = labelMat.T
normDataMat, meanData, stdData = featureNormalize(dataMat)
alpha = 0.01
theta = gradientDescent(normDataMat, labelMat, alpha)
print theta
testX = [1, 45, 85]
testX = mat(testX)
normTestX = (testX - meanData) / stdData
print sigmoid(normTestX * theta)

count = 0
tmp = sigmoid(normDataMat * theta)
m = shape(labelMat)[0]
for i in range(m):
	if round(tmp[i, 0])== labelMat[i, 0]:
		count += 1
print count

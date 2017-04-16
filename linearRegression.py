from numpy import *

def loadDataSet(filename):
	numFeat = len(open(filename).readline().strip().split(',')) - 1
	dataSet = []; labelSet = []
	fr = open(filename)
	for line in fr.readlines():
		lineArr = []
		lineArr.append(1.0)
		curLine = line.strip().split(',')
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

def gradientDescent(dataMat, labelMat, alpha, num_iters = 10000):
	m = shape(dataMat)[0]
	numFeat = shape(dataMat)[1]
	theta = mat(zeros((numFeat, 1)))
	for i in range(num_iters):
		tmp = dataMat * theta - labelMat
		tmp = dataMat.T * tmp
		theta = theta - alpha * tmp / m
	return theta

def normalEquation(dataMat, labelMat):
	return (dataMat.T * dataMat).I * dataMat.T * labelMat

dataSet, labelSet = loadDataSet('D:\\machine-learning-ex1\\ex1\\ex1data2.txt')
dataMat = mat(dataSet)
labelMat = mat(labelSet)
labelMat = labelMat.T
normDataMat, meanData, stdData = featureNormalize(dataMat)
alpha = 0.01
theta = gradientDescent(normDataMat, labelMat, alpha)
print theta
testX = [1, 1650, 3]
testX = mat(testX)
normTestX = (testX - meanData) / stdData
print normTestX * theta
thetaE = normalEquation(dataMat, labelMat)
print thetaE
print testX * thetaE

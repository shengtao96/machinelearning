#coding:utf-8
from numpy import *

def loadDataSet(filename):
	fr = open(filename)
	numFeat = len(fr.readline().strip().split(',')) - 1
	dataSet = []; labelSet = []
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
	m, n = shape(dataMat)
	theta = mat(zeros((n, 1)))
	for i in range(num_iters):
		tmp = dataMat * theta - labelMat
		tmp = dataMat.T * tmp
		theta = theta - alpha * tmp #这里不除以m，是因为没有必要，alpha可以直接作为步长来进行收敛，除以m的话有点多此一举
	return theta

def equation(dataMat, labelMat):
	return (dataMat.T * dataMat).I * dataMat.T * labelMat

dataSet, labelSet = loadDataSet('D:\\machine-learning-ex1\\ex1\\ex1data2.txt')
dataMat = mat(dataSet); labelMat = mat(labelSet).T
normDataMat, meanData, stdData = featureNormalize(dataMat)
alpha = 0.01
theta = gradientDescent(normDataMat, labelMat, alpha)
print theta
testX = [1, 1650, 3]
testX = mat(testX)
normTestX = (testX - meanData) / stdData
print normTestX * theta

thetaE = equation(dataMat, labelMat)
print thetaE
print testX * thetaE

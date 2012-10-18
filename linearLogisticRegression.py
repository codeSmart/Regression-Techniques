#Author: Krishna Sai P B V
#Date: 10/08/2012

#Input to this file is nothing
#Output is linear and logistic regression techniques running
#until obtaining convergence (displays error rate)
#For the convergent value, calculates AUC (Area under the ROC curve)

#!/usr/bin python

import string,re,sys,math, operator, random
from decimal import *
from collections import defaultdict

spamDataSet=open('spambase.data','r')
docValues=open('documentationValues.py','r')
perceptronData=open('perceptronData.txt','r')
flagValues=docValues.readlines()
spamData=spamDataSet.readlines()
perceptronValues=perceptronData.readlines()
#number of folds.
k=10
totalEmails=0
#helper array to calculate feature mean
totalSum=[]
kFold = [[]*k for x in xrange(k)]
zScore = [[]*k for x in xrange(k)]
meanValues=[]
sDeviation=[]
setSpamNSpam=defaultdict(list)

#helper array to caluclate mean values
for i in range(0, 58):
	totalSum.append(0)
	sDeviation.append(0)

#calculating kFold by parsing through
#the given spam data file.
i=0
for line in spamData:
	j=i%10;
	frNo=0
	kFold[j].append(line)
	line=line.split(',')
	toAppend= line[57].strip('\n').strip('\r')
	for feature in line:
		totalSum[frNo]+=float(feature)
		frNo+=1
	i=i+1
	totalEmails+=1

#mean values calculation
for i in range(0,len(totalSum)):
	meanValues.append(float(totalSum[i]) / float(totalEmails))

#calculating standard deviations
for fold in kFold:
	for email in fold:
		email=email.split(',')
		frNo=0
		for feature in email:
			sDeviation[frNo]+= math.pow((float(feature) - meanValues[frNo]),2)	
			frNo+=1

# standard deviation helpers
for i in range(0,len(sDeviation)):
		sDeviation[i]= math.sqrt(float(sDeviation[i]) / float(totalEmails-1))

#calculating zScores
for i in range(0,len(kFold)):
	for j in range(0,len(kFold[i])):
		tempArray=kFold[i][j].split(',')
		flagArray=[]
		flagArray2=[]
		for k in range(0,len(tempArray)-1):
			temp=tempArray[k].rstrip('\n').rstrip('\r')
			temp1=float(float(temp) - meanValues[k]) / float(sDeviation[k])			
			flagArray.append(temp1)
		flagArray.append(float(tempArray[57].strip('\n').strip('\r')))
		for feature in flagArray:
			flagArray2.append(feature)
		zScore[i].append(flagArray2)

trainingSet=[]
testingSet=[]
for i in range(0,len(zScore)):
	if(i!=1):
		for k in range(0,len(zScore[i])):
			trainingSet.append(zScore[i][k])
	else:
		for k in range(0,len(zScore[i])):
			testingSet.append(zScore[i][k])

random.shuffle(trainingSet)

#calculating AUC
def calAUC(fpr, tpr):
	sumValue=0
	for i in range(1,len(fpr)):
		sumValue+= ((fpr[i]-fpr[i-1]) * (tpr[i] + tpr[i-1]))
	print "AUC: ",(0.5*sumValue)
	#return (0.5*sumValue)


#calculate scores and FPR and TPR for ROC graph
def calROC(weights,flag):
	scores=[]
	if (flag==0):
		for i in range(0,len(testingSet)):
			hValue=0.0
			for k in range(0,len(weights)-1):
				if k==0:
					hValue+= float(weights[k] * 1.0)
				else:
					hValue+= (float(weights[k]) * float(testingSet[i][k-1]))
			scores.append(hValue)
	else:
		for i in range(0,len(testingSet)):
			hValue=0.0
			for k in range(0,len(weights)-1):
				if k==0:
					hValue+= float(weights[k] * 1.0)
				else:
					hValue+= (float(weights[k]) * float(testingSet[i][k-1]))
			hValue = 1.0 / float(1 + math.exp(0-hValue))
			scores.append(hValue)
	#print scores
	fpr=[]
	tpr=[]
	tnr=[]
	fnr=[]
	scoresSorted=sorted(scores,reverse=True)
	for i in range(0,len(scoresSorted)):
		fp=0
		tp=0
		tn=0
		fn=0
		for k in range(0,len(scoresSorted)):
			if(i!=k):
				if(scores[k] >scoresSorted[i]):
					if(testingSet[k][57]==0.0):
						fp=fp+1
					else:
						tp=tp+1
				else:
					if(testingSet[k][57]==1.0):
						fn=fn+1
					else:
						tn=tn+1
		tpr.append(tp)		
		fpr.append(fp)	
		fnr.append(fn)
		tnr.append(tn)
	for i in range(0,len(fpr)):
		fpr[i]=(float(fpr[i])/float(fpr[i] + tnr[i]))
		tpr[i]=(float(tpr[i])/float(tpr[i] + fnr[i]))
		#print fpr[i], tpr[i]
	calAUC(fpr,tpr)

#checking if convergence is acheived
def checkForConvergence(errorValues):
	if(abs(errorValues[len(errorValues)-2] - errorValues[len(errorValues)-1])<0.001):
		return True
	else:
		return False


stochasticWeights=[[0]*59 for x in xrange(len(trainingSet)+1)]
lambdaConstant=0.001
def calErrorFunctionValue(iterationCount):
	l=0
	sumValueForErrorFunction=0
	for i in range(0,len(trainingSet)):
		hValue=0.0
		for k in range(0,len(stochasticWeights[i+1])-1):
			if k==0:
				hValue+= float(stochasticWeights[i+1][k] * 1.0)
			else:
				hValue+= (float(stochasticWeights[i+1][k]) * float(trainingSet[i][k-1]))
		flag=(hValue - trainingSet[i][57])
		flag=math.pow(flag,2)
		sumValueForErrorFunction+= flag
	print iterationCount,math.sqrt(float(sumValueForErrorFunction) / float(4141))
	return math.sqrt(float(sumValueForErrorFunction) / float(4141))

stochasticWeightsPrevious=[]

#calculating stochastic weights. At each iteration, the last updated feature value will be
#0th stochastic weights for next iteration
def updateStochasticWeights():
	sumValueForErrorFunction=0
	notContinue=False
	rmse=[]
	localLambda=lambdaConstant;
	iterationCount=0
	while(not notContinue):
		for i in range(0,len(trainingSet)):
			hValue=0.0
			for k in range(0,len(stochasticWeights[i])-1):
				if k==0:
					hValue+= float(stochasticWeights[i][0] * 1.0)
				else:
					hValue+= float(stochasticWeights[i][k]) * float(trainingSet[i][k-1])
			for k in range(0,len(stochasticWeights[i])-1):
				if (k == 0):
					flag=localLambda * (hValue-trainingSet[i][57])
					stochasticWeights[i+1][0]= stochasticWeights[i][0] - (flag * 1.0)
				else:
					flag=localLambda * (hValue-trainingSet[i][57])
					stochasticWeights[i+1][k]= stochasticWeights[i][k] - (flag * float(trainingSet[i][k-1]))
			stochasticWeights[i+1][58]= trainingSet[i][57]
			stochasticWeights[0]=stochasticWeights[len(stochasticWeights)-1]
		rmse.append(calErrorFunctionValue(iterationCount))
		#stochasticWeightsPrevious.append(stochasticWeights[0])
		if(iterationCount>=1):
			notContinue=checkForConvergence(rmse)
		if(notContinue==True):
			calROC(stochasticWeights[len(stochasticWeights)-1],0)
		#else:
		#	localLambda=float(localLambda /10.0);
		#	stochasticWeights[0]=stochasticWeightsPrevious[len(stochasticWeightsPrevious)-2]
		iterationCount+=1

batchWeights=[0]*58
#batchWeightsPrevious=[]

#calculate error function value for batch gradient descent
def calErrorFunctionValueForBatch(iterationCount):
	sumValueForErrorFunction=0
	for i in range(0,len(trainingSet)):
		hValue=0.0
		for k in range(0,len(trainingSet[i])):
			if k==0:
				hValue+= float(batchWeights[k] * 1.0)
			else:
				hValue+= (float(batchWeights[k]) * float(trainingSet[i][k-1]))
		flag=(hValue - trainingSet[i][57])
		flag=math.pow(flag,2)
		sumValueForErrorFunction+= flag
	print iterationCount,math.sqrt(float(sumValueForErrorFunction) / float(4141.0))
	return math.sqrt(float(sumValueForErrorFunction) / float(4141.0))

#calculate batch weights for linear regression
def updateBatchWeights(batchWeights):
	rmse=[]
	notContinue=False
	localLambda=0.01
	iterationCount=0
	while(not notContinue):
		batchWeightsPrevious=[]
		batchWeightsPrevious.append(batchWeights)
		for i in range(0,len(batchWeights)):
			sumValue=0.0
			for k in range(0,len(trainingSet)):
				hValue=0.0
				for f in range(0,len(batchWeights)):					
					if(f==0):
						hValue+= batchWeights[0] * 1.0			
					else:
						hValue+= float(batchWeights[f]) * trainingSet[k][f-1]
				diffValue=hValue - trainingSet[k][57]
				if (i==0):
					prodValue = diffValue * 1.0
				else:
					prodValue=diffValue * trainingSet[k][i-1]
					sumValue+=prodValue
			batchWeights[i]= batchWeights[i] - (float((localLambda * sumValue))/float(4141.0))
		rmse.append(calErrorFunctionValueForBatch(iterationCount))
		batchWeightsPrevious.append(batchWeights)
		if(iterationCount>1):
			notContinue=checkForConvergence(rmse)
		if(notContinue==True):
			calROC(batchWeights,0)
		iterationCount+=1

stochasticWeightsLogistic=[[0]*59 for x in xrange(len(trainingSet)+1)]
#lambdaConstant=0.01
#calculating error function value for stochastic gradient using logistic regression
def calErrorFunctionValueForLogisticStochastic(iterationCount):
	sumValueForErrorFunction=0
	for i in range(0,len(trainingSet)):
		hValue=0.0
		for k in range(0,len(stochasticWeightsLogistic[i+1])-1):
			if k==0:
				hValue+= float(stochasticWeightsLogistic[i+1][k] * 1.0)
			else:
				hValue+= (float(stochasticWeightsLogistic[i+1][k]) * float(trainingSet[i][k-1]))
		hValue = 1.0 / float(1 + math.exp(0-hValue))
		flag=(hValue - trainingSet[i][57])
		flag=math.pow(flag,2)
		sumValueForErrorFunction+= flag
	print iterationCount,math.sqrt(float(sumValueForErrorFunction) / float(4141.0))
	return math.sqrt(float(sumValueForErrorFunction) / float(4141.0))


#calculate logistic regression for stochastic regression
def updateStochasticWeightsLogistic():
	sumValueForErrorFunction=0
	rmse=[]
	notContinue=False
	iterationCount=0
	lambdaConstant=1.0
	while(not notContinue):	
		for i in range(0,len(trainingSet)):
			hValue=0.0
			for k in range(0,len(stochasticWeightsLogistic[i])-1):
				if k==0:
					hValue+= float(stochasticWeightsLogistic[i][0] * 1.0)
				else:
					hValue+= float(stochasticWeightsLogistic[i][k]) * float(trainingSet[i][k-1])
			hValue = 1.0 / float(1 + math.exp(0-hValue))
			for k in range(0,len(stochasticWeightsLogistic[i])-1):
				if (k == 0):
					flag=lambdaConstant * (hValue-trainingSet[i][57])* hValue * (1.0-hValue) 
					stochasticWeightsLogistic[i+1][0]= stochasticWeightsLogistic[i][0] - (flag * 1.0)
				else:
					flag=lambdaConstant * (hValue-trainingSet[i][57]) * hValue * (1.0-hValue)
					stochasticWeightsLogistic[i+1][k]= stochasticWeightsLogistic[i][k] - (flag * float(trainingSet[i][k-1]))
			stochasticWeightsLogistic[i+1][58]= trainingSet[i][57]
			stochasticWeightsLogistic[0]=stochasticWeightsLogistic[len(stochasticWeightsLogistic)-1]
		rmse.append(calErrorFunctionValueForLogisticStochastic(iterationCount))
		if(iterationCount>1):
			notContinue=checkForConvergence(rmse)
		if(notContinue==True):
			calROC(stochasticWeightsLogistic[len(stochasticWeights)-1],1)
		iterationCount+=1

batchWeightsLogistic=[0]*58


#calculate error function value for batch gradient using logistic regression
def calErrorFunctionValueForLogisticBatch(iterationCount):
	sumValueForErrorFunction=0
	for i in range(0,len(trainingSet)):
		hValue=0.0
		for k in range(0,len(trainingSet[i])):
			if k==0:
				hValue+= float(batchWeightsLogistic[k] * 1.0)
			else:
				hValue+= (float(batchWeightsLogistic[k]) * float(trainingSet[i][k-1]))
		hValue = 1.0 / float(1+math.exp(0-hValue))
		flag=(hValue - trainingSet[i][57])
		flag=math.pow(flag,2)
		sumValueForErrorFunction+= flag
	print iterationCount,math.sqrt(float(sumValueForErrorFunction) / float(4141.0))
	return math.sqrt(float(sumValueForErrorFunction) / float(4141.0))

#update batch weights using logistic regression.
def updateBatchWeightsLogistic():
	rmse=[]
	notContinue=False
	iterationCount=0
	lambdaConstant=0.01
	while(not notContinue):
		for i in range(0,len(batchWeightsLogistic)):
			sumValue=0.0
			for k in range(0,len(trainingSet)):
				hValue=0.0
				for f in range(0,len(batchWeightsLogistic)):					
					if(f==0):
						hValue+= batchWeightsLogistic[0] * 1.0			
					else:
						hValue+= float(batchWeightsLogistic[f]) * trainingSet[k][f-1]
				hValue = 1.0 / float(1+math.exp(0-hValue))
				diffValue=hValue - trainingSet[k][57]
				if (i==0):
					prodValue = diffValue * 1.0
				else:
					prodValue=diffValue * trainingSet[k][i-1]
				sumValue+=prodValue
			batchWeightsLogistic[i]= batchWeightsLogistic[i] - (float((lambdaConstant * sumValue * hValue * (1-hValue)))/float(4141.0))
		rmse.append(calErrorFunctionValueForLogisticBatch(iterationCount))
		if(iterationCount>=1):
			notContinue=checkForConvergence(rmse)
		if(notContinue):
			calROC(batchWeightsLogistic,1)
		iterationCount+=1

#data strucuture to store the perceptron data
perceptronFeatures=[[]*1000 for x in xrange(1000)]

for i in range(0,len(perceptronFeatures)):
	for k in range(0,5):
		perceptronFeatures[i].append(0.0)

#parse perceptron data and get features
def getPerceptronFeatures():
	i=0
	for dataPoint in perceptronValues:
		dataPoint=dataPoint.split('\t')
		k=0
		for feature in dataPoint:
			feature=float(feature.strip('\n'))
			perceptronFeatures[i][k]=feature
			k+=1
		i+=1
	return perceptronFeatures

perceptronWeights=[0.0]*5

#perceptron learning algorithm
def perceptronLearning():
	totalMistakes=[]
	for iteration in range(0,10):
		totalMistakesNow=0
		perceptronFeatures=getPerceptronFeatures()			
		for i in range(len(perceptronFeatures)):
			hypothesisValue=0.0
			featureZero=1.0
			if(float(perceptronFeatures[i][4])==-1):
				featureZero=-1.0
				for k in range(0,len(perceptronFeatures[i])-1):
					perceptronFeatures[i][k] = float(-1.0 * float(perceptronFeatures[i][k]))
			for k in range(0,len(perceptronFeatures[i])):	
				if(k==0):
					hypothesisValue+= perceptronWeights[k] * featureZero		
				else:
					#print i,k, len(perceptronFeatures[i])
					hypothesisValue+= perceptronWeights[k] * float(perceptronFeatures[i][k-1])
			if(hypothesisValue<=0):
				totalMistakesNow+=1
				for k in range(0,len(perceptronWeights)):
					if(k==0):
						perceptronWeights[k] = perceptronWeights[k] + featureZero
					else:
						perceptronWeights[k] = perceptronWeights[k] + float(perceptronFeatures[i][k-1])
		totalMistakes.append(totalMistakesNow)
		if(iteration>1):
			if(totalMistakes[iteration-1]==totalMistakesNow):
				normalizedWeights=[]
				print "Classifier weights: ", perceptronWeights
				for w in range(1,len(perceptronWeights)):
					normalizedWeights.append(float(perceptronWeights[w])/float(0-perceptronWeights[0]))
				print "Normalized with threshold:" ,normalizedWeights
				break;			
		print "Iteration:",iteration," total mistakes:", sum(totalMistakes)
		
		
#execution of all the gradient descent problems			
def main():
	print "***** Linear Stochastic Gradient Descent *****"
	updateStochasticWeights()
	print "\n"	
	print "***** Linear Batch Gradient Descent *****"
	updateBatchWeights(batchWeights)
	print "\n"
	print "***** Logisitic Stochastic Gradient Descent *****"
	updateStochasticWeightsLogistic()
	print "\n"
	print "***** Logisitc Batch Gradient Descent *****"
	updateBatchWeightsLogistic()
	print "\n"
	print "***** Perceptron Learning *****"
	perceptronLearning()
	print "\n"

if __name__ == "__main__":
	main()

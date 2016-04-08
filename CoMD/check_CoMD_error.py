#!/usr/bin/python
import sys
import math

goldFile = "output_CoMD_01_01_01.txt"

fileToCompare = sys.argv[1]
print fileToCompare

fp1 = open(goldFile, 'r')
fp2 = open(fileToCompare, 'r')

goldValues = []
approxValues = []

for line in fp1.readlines():
	line = line.strip()
	fields = line.split(",")
	val1 = float(fields[1].strip())
	val2 = float(fields[2].strip())
	val3 = float(fields[3].strip())
	val4 = float(fields[4].strip())
	val5 = float(fields[5].strip())
	vals = [val1,val2,val3,val4,val5]
	goldValues.append(vals)

for line in fp2.readlines():
	line = line.strip()
	fields = line.split(",")
	val1 = float(fields[1].strip())
	val2 = float(fields[2].strip())
	val3 = float(fields[3].strip())
	val4 = float(fields[4].strip())
	val5 = float(fields[5].strip())
	vals = [val1,val2,val3,val4,val5]
	approxValues.append(vals)

totalDiff = 0.0
numElems = len(goldValues)
count = 0
for i in range(numElems):
	goldVals = goldValues[i]
        approxVals = approxValues[i]
	for k in range(5):
		goldVal = goldVals[k]
		approxVal = approxVals[k]
        	if(math.fabs(goldVal) < 0.00001):
			goldVal = 0
		if(goldVal == 0):
			continue
		diff = math.fabs((goldVal - approxVal)/float(goldVal))
        	#print str(diff) + "   " + str(i) + "    " + str(goldValues[i]) + "    " + str(approxValues[i])
		totalDiff = totalDiff + diff
		count = count + 1

avgDiff = totalDiff/float(numElems*5)
avgDiffCount = totalDiff/float(count)

print "Approximation error : " + str(avgDiff) + " :  error based on count : " + str(avgDiffCount)

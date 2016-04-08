#!/usr/bin/python
import sys
import math

goldFile = "lulesh_dump_1_1_1.txt"

fileToCompare = sys.argv[1]
print fileToCompare

fp1 = open(goldFile, 'r')
fp2 = open(fileToCompare, 'r')

goldValues = []
approxValues = []

for line in fp1.readlines():
	line = line.strip()
	fields = line.split("=")
	val = float(fields[1].strip())
	goldValues.append(val)

for line in fp2.readlines():
	line = line.strip()
	fields = line.split("=")
	val = float(fields[1].strip())
	approxValues.append(val)

totalDiff = 0.0
numElems = len(goldValues)
count = 0
for i in range(numElems):
	goldVal = goldValues[i]
        approxVal = approxValues[i]
        if(math.fabs(goldVal) < 0.0001):
		goldVal = 0
	if(goldVal == 0):
		continue
	diff = math.fabs((goldValues[i] - approxValues[i])/float(goldValues[i]))
        #print str(diff) + "   " + str(i) + "    " + str(goldValues[i]) + "    " + str(approxValues[i])
	totalDiff = totalDiff + diff
	count = count + 1

avgDiff = totalDiff/float(numElems)
avgDiffCount = totalDiff/float(count)

print "Approximation error : " + str(avgDiff) + " :  error based on count : " + str(avgDiffCount)

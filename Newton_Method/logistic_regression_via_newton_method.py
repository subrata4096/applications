#!/usr/bin/python
import numpy as np
import sys
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import LeavePOut
from scipy.stats import *
from math import *

tolerance = 0.0001
maxsteps = 100000

degree1 = 1
degree2 = 1


#Black-Scholes model is the calculation of implied volatilities.
#http://www.stat.ucla.edu/~nchristo/statistics_c183_c283/statc183c283_implied_volatility.pdf
S=21.0
E=20.0
r=0.1
t=0.25
C=1.875
def d1(sigma):
	#print sigma
	return (np.log(S/E)+(r+0.5*pow(sigma,2))*t)/(sigma*sqrt(t))

def d2(sigma):
	return d1(sigma) - sigma*sqrt(t)

def f(sigma):
	a = d1(sigma)
	b = d2(sigma)
	return S*norm.pdf(a) - E*np.exp(-r*t)*norm.pdf(b) - C

def df(sigma):
	d11 = (pow(sigma,2)*t*sqrt(t)-(np.log(S/E)+(r+pow(sigma,2)/2)*t)*sqrt(t))/(pow(sigma,2)*t)
	d22 = d11-sqrt(t)
	val = S*norm.pdf(d1(sigma))*d11-E*np.exp(-r*t)*norm.pdf(d2(sigma))*d22
	#print d11, d22, val
	if(val == 0):
		print "error", sigma, d2(sigma)
		exit(0)
	return val

def newtonRap(cp, price, s, k, t, rf):
    v = sqrt(2*pi/t)*price/s
    print "initial volatility: ",v
    for i in range(1, 100):
        d1 = (log(s/k)+(rf+0.5*pow(v,2))*t)/(v*sqrt(t))
        d2 = d1 - v*sqrt(t)
        vega = s*norm.pdf(d1)*sqrt(t)
        price0 = cp*s*norm.cdf(cp*d1) - cp*k*exp(-rf*t)*norm.cdf(cp*d2)
        v = v - (price0 - price)/vega
        print "price, vega, volatility\n",(price0, vega, v)
        if abs(price0 - price) < 1e-25 :
            break
    return v

def PHI(t):
	#this is the actual logistics regression function
	#if(t>0):	
	#	return 1.0/(1.0 + np.exp(-t))
	#else:
	#	return np.exp(t)/(1.0 + np.exp(t))
	#Transcendental equation
	return float(t - np.exp(-t))

def dPHI(t):
	#return float(PHI(t)*(1-PHI(t)))
	#Transcendental equation
	return float(1+np.exp(-t))

#P = 300.0
#a = 1.36
#b = 0.003183
#T = 300.0
#R = 0.0820578
#def F(x):
	#return 2*x*x - 1
	#return P*pow(x,3) - (P*b + R*T)*pow(x,2) + a*x - a*b
#	return (P + a/pow(x,2))*(x - b) - R*T

#def dF(x):
	#return 4*x
#	return P - (a/pow(x,2)) + (2*a*b)/pow(x,3)

def F(x):
	#return pow(x,6) + 4*pow(x,5) + 3*pow(x,4) + 2*pow(x,2) + x - 18372.33
	return np.exp(x) + x - 33.782379
def dF(x):
	#return 6*pow(x,5) + 20*pow(x,4) + 12*pow(x,3) + 4*x + 1
	return np.exp(x) + 1

def getEstimatorthe_f(deg):
	inArr = []
	outArr = []
	for num in np.arange(1,10,0.01):
		inArr.append([num])
		out = F(num)
		outArr.append([out])
	#end for
	#print inArr         
        #print outArr        
        inArr = np.array(inArr)
        outArr = np.array(outArr)
        #outArr = np.transpose(np.array(outArr))
        #inArr = np.array([[0, 0], [1,11], [2,12],[3,13]])
        #outArr = np.array([0,3,6,12])
        #print inArr.shape   
        #print outArr.shape 
	polyReg = Pipeline([('poly', PolynomialFeatures(degree=deg)),('linear', LinearRegression(normalize=True))]) 
	polyReg.fit(inArr,outArr)
	print polyReg.score(inArr,outArr)
	return polyReg

def getEstimatorthe_df(deg):
	inArr = []
	outArr = []
	for num in np.arange(1,10,0.01):
		inArr.append([num])
		out = dF(num)
		outArr.append([out])
	#end for
	#print inArr
	#print outArr
	inArr = np.array(inArr)
	outArr = np.array(outArr)
	#outArr = np.transpose(np.array(outArr))
	#inArr = np.array([[0, 0], [1,11], [2,12],[3,13]])
        #outArr = np.array([0,3,6,12])
	#print inArr.shape
	#print outArr.shape
	polyReg = Pipeline([('poly', PolynomialFeatures(degree=deg)),('linear', LinearRegression(normalize=True))]) 
	polyReg.fit(inArr,outArr)
	print polyReg.score(inArr,outArr)
	return polyReg

def newton(xs, actual):
	
	estm_f = getEstimatorthe_f(degree1)
	estm_df = getEstimatorthe_df(degree2)

        x = 0.0
	xprim = 0.0
        t1 = 0.0
	t2 = 0.0
        x = xs;
        xprim = xs + 2*tolerance;
        iterCount = 0;
        while((abs(x - xprim)  >= tolerance)  and (iterCount < maxsteps)):
                xprim = x
		if(actual == True):
                	t1 = F(x)
                	t2 = dF(x)
		else:
                	t1 = estm_f.predict(x)
                	t2 = estm_df.predict(x)

                #print "x= ", x, " t1= ", t1, " t2= ", t2
                x = x  - float(t1 / t2)
                iterCount = iterCount + 1
        #end while

        if((abs(x  - xprim) >=  tolerance)):
                #x = INFTY;
                x = 999999
	#end if        

        print "Newton's method result " + str(x) + ", IterCount = " + str(iterCount)
        return x


if __name__ == "__main__":
	degree1 = int(sys.argv[1])
	degree2 = int(sys.argv[2])

	xs = 0.1
	#print f(11.4)
	#print df(11.4)
        x1 = newton(xs,True)
        x2 = newton(xs,False)
	print F(x2)
       	#err = float(abs(x1 - x2))/float(x1)
	#print "degree1= ", degree1, " degree2= ", degree2, " err= ", err 

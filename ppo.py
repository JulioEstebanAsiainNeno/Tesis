import tensorflow as tf

#input
#Pass an action to the cpp, and the cpp passes velocity, position, distance to goal, rewards

LAMBDA = 0.95
GAMMA = 0.99
Tmax = 8000
Ephi = 20
BETA = 1.0
KLtarget = 0.0015
XI = 50.0
lrtheta1 = 0.00005
lrtheta2 = 0.00002
Ev = 10
lrtheta3 = 0.001
BETAhigh = 2.0
ALPHA = 1.5
BETAlow = 0.5

#HYPERPARAMETERS HYBRID
Rsafe = 0.1
Rrisk = 0.8
Pscale = 1.25
Vmax = 0.5

#Initialize Policy "policy0" and Value function "value". Theta = policy parameters

iterations = 100
numRobots = 2

#T[i] = cantidad de frames ejecutados por robot

maxt = -10000000
sumA = 0
sumT = 0
sumL = 0
sumK = 0
breakIT = false
for it in range(iterations):
	#Collect data
	for i in range(numRobots):
		#while(1):
			#Run policy for F frames
			#if o = goal 
				#break
		#T[i] = F
		sumT = sumT + T[i]
		for t in range(T[i]):
			for l in range(T[i])
				delta[i][t] = reward[i][t] + GAMMA*value(s[i][t+1]) - value(s[i][t])
				sumA = sumA + ((GAMMA*LAMBDA)^l)(delta[i][t])
			A[i][t] = sumA
		if sumT > Tmax:
			break
	policyOld = policyNew
	#//Update policy
	for j in range(Ephi):
		for i in range(numRobots):
			for t in range T[i]:
				sumL = sumL + (policyOld(parameters)/policyNew(parameters)*A[i][t]) - BETA*KL(parameters) + XI*(max(0, KL(parameters) - 2*KLtarget))^2
				Lppo(theta) = sumL
		if KL(parameters) > 4*KLtarget:
			breakIT = true
			break
	if breakIT == true:
		breakIT = false
		continue
		#Update theta wih lrtheta Lppo(theta)
	#//Update value function
	for k in range(Ev):
		for i in range(numRobots):
			for t in range(T[i]):
				if t > maxt:
					maxt = t
				for a in range(maxt, t, -1):
					sumK = -1*(sumK + (GAMMA^(maxt-t))*reward[i][maxt] - (value(s[i][t]))^2)
		Lv(theta) = sumK
		#Update there with lrtheta Lv(theta)
	#//Adapt KL Penalty Coefficient
	if KL(parameters) > BETAhigh*KLtarget:
		BETA = ALPHA*BETA
	else if KL(parameters) < BETAlow*KLtarget:
		BETA = BETA/ALPHA
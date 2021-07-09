#!/usr/bin/env python

import rvo2
import numpy as np
import math
import keras
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls
import pygame
import time

#Global variables
num_agents = 4
reward_arrival = 15.0
wg = 2.5
reward_collide = -15.0
ww = -0.1
time_step = 1/10.
NEIGHBOR_DIST = 1.5
max_neighbors = 5
time_horizon_obst = 2
radius = 0.4
max_speed = 1.0
num_neighbors = 10 #number of neighbors that each agent considers
neighbor_distance = 15.0 #max distance that agents can perceive neighbors
time_horizon_ORCA = 5.0 #time horizon to determine collisions with other agents
time_horizon_obst_ORCA = 1.3 # time horizon to determine collisions with obstacles
radius_ORCA = 0.5 #distance that the agents want to keep from other agents
max_speed_ORCA = 1.0 #maximum speed that agents can move with

#PPO Parameters
LAMBDA = 0.95
GAMMA = 0.99
Tmax = 8000
Ephi = 20
BETA = 1.0
KLtarget = 0.0015
XI = 50.0
Ev = 10
BETAhigh = 2.0
ALPHA = 1.5
BETAlow = 0.5

display_width = 400
display_height = 300

def getDistanceToPoint(point1, point2):
	distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
	return distance

def getDistanceToGoal(agent):
	distance_to_goal_x = sim.getAgentGoal(agent)[0] - sim.getAgentPosition(agent)[0]
	distance_to_goal_y = sim.getAgentGoal(agent)[1] - sim.getAgentPosition(agent)[1]
	distance_to_goal = (abs(distance_to_goal_x), abs(distance_to_goal_y))
	return distance_to_goal 

def getDistanceToObstacle(agent):
	distance_to_obstacle1 = abs(obstacle1_y - sim.getAgentPosition(agent)[1])
	distance_to_obstacle2 = abs(obstacle2_y - sim.getAgentPosition(agent)[1])
	if distance_to_obstacle1 >= distance_to_obstacle2:
		return distance_to_obstacle1
	else:
		return distance_to_obstacle2 

def getNeighborPositions(agent):
	my_position = sim.getAgentPosition(agent)
	neighbor_list = []
	num_neighbors = sim.getAgentNumAgentNeighbors(agent)
	neighbor_positions = [(0.0, 0.0) for x in range(len(agent_list)-1)]

	for neighbor in range(num_neighbors):
		neighbor_list.append(sim.getAgentAgentNeighbor(agent, neighbor))

	for neighbor in range(len(neighbor_list)):
		neighbor_positions[neighbor] = sim.getAgentPosition(sim.getAgentAgentNeighbor(agent, neighbor))

	return neighbor_positions

def getRelativeNeighborVelocityList(agent):
	my_velocity = sim.getAgentVelocity(agent)
	neighbor_list = []
	num_neighbors = sim.getAgentNumAgentNeighbors(agent)
	neighbor_velocities = []
	relative_neighbor_velocities = [(0.0, 0.0) for x in range(len(agent_list)-1)]

	for neighbor in range(num_neighbors):
		neighbor_list.append(sim.getAgentAgentNeighbor(agent, neighbor))

	for neighbor in range(len(neighbor_list)):
		neighbor_velocities.append(sim.getAgentVelocity(sim.getAgentAgentNeighbor(agent, neighbor))) #global velocities
		relative_neighbor_velocity_x = my_velocity[0] - sim.getAgentVelocity(sim.getAgentAgentNeighbor(agent, neighbor))[0]
		relative_neighbor_velocity_y = my_velocity[1] - sim.getAgentVelocity(sim.getAgentAgentNeighbor(agent, neighbor))[1]
		relative_neighbor_velocities[neighbor] = ((relative_neighbor_velocity_x, relative_neighbor_velocity_y))
	return relative_neighbor_velocities

def getDistanceToNeighbor(agent, neighbor):
	my_position = sim.getAgentPosition(agent)
	neighbor_position = sim.getAgentPosition(sim.getAgentAgentNeighbor(agent, neighbor))
	distance_to_neighbor = getDistanceToPoint(my_position, neighbor_position)
	return distance_to_neighbor

def separateVelocities(velocity):
	translational_velocity = math.sqrt(velocity[0]**2 + velocity[1]**2)
	if velocity[0] > 0:
		angular_velocity = math.atan(velocity[1]/velocity[0])
	elif velocity[0] < 0 and velocity[1] >= 0:
		angular_velocity = math.atan(velocity[1]/velocity[0]) + math.pi
	elif velocity[0] < 0 and velocity[1] < 0:
		angular_velocity = math.atan(velocity[1]/velocity[0]) - math.pi
	elif velocity[0] == 0 and velocity[1] > 0:
		angular_velocity = math.pi/2
	elif velocity[0] == 0 and velocity[1] < 0:
		angular_velocity = -math.pi/2
	elif velocity[0] == 0 and velocity[1] == 0:
		angular_velocity = 0
	
	return translational_velocity, angular_velocity

def getReward(agent, goal_position_difference):
	reward_collision = getCollisionReward(agent)
	reward_goal = getGoalReward(agent, goal_position_difference)
	reward_movement = getMovementReward(agent)
	reward_total = reward_collision + reward_goal + reward_movement
	print ("reward_collision: ", reward_collision)
	print ("reward_goal: ", reward_goal)
	print ("reward_movement: ", reward_movement)
	print ("reward_total: ", reward_total)
	return reward_total

def getCollisionReward(agent):
	my_position = sim.getAgentPosition(agent)
	neighbor_position_list = getNeighborPositions(agent)
	obstacle_distance = getDistanceToObstacle(agent)
	listpos = 0
	for position_neighbor in neighbor_position_list:
		distance_neighbor = math.sqrt(((my_position[0] - position_neighbor[0])**2 + (my_position[1] - position_neighbor[1])**2))
		#print ("Distance to neighbor ", neighbor, ": ", neighbor_distance)
		if (distance_neighbor < 2*radius or obstacle_distance < radius) and getDistanceToNeighbor(agent, listpos) != 0:
			collision_reward = reward_collide
			listpos += listpos + 1
			print ("Agent ", agent, "collided")
			time.sleep(5)
			break
		else:
			collision_reward = 0
			listpos += listpos +1
	return collision_reward

def getMovementReward(agent):
	_, angular_velocity = separateVelocities(sim.getAgentVelocity(agent))
	if abs(angular_velocity) > 0.7:
		movement_reward =  ww*abs(angular_velocity)
	else:
		movement_reward = 0

	return movement_reward

def getGoalReward(agent, goal_position_difference):
	goal_reward = 0
	goal = sim.getAgentGoal(agent)
	agent_position = sim.getAgentPosition(agent)
	distance_to_goal = math.sqrt(((agent_position[0] - goal[0])**2 + (agent_position[1] - goal[1])**2))
	print ("distance_to_goal: ", distance_to_goal)
	if distance_to_goal < 0.1:
		goal_reward = reward_arrival
		done = True
	else:
		goal_reward = wg*goal_position_difference
		done = False

	return goal_reward

def reset():
	state = [0 for x in range(num_agents)]
	done = False
	'''
	sim.setAgentPosition(0, ((-20.0, 30.0)))
	sim.setAgentPosition(1, ((-17.0, 30.0)))
	sim.setAgentPosition(2, ((-14.0, 30.0)))
	sim.setAgentPosition(3, ((-20.0, 29.0)))
	sim.setAgentPosition(4, ((-17.0, 29.0)))
	sim.setAgentPosition(5, ((-14.0, 29.0)))
	sim.setAgentPosition(6, ((-20.0, 28.0)))
	sim.setAgentPosition(7, ((-17.0, 28.0)))
	sim.setAgentPosition(8, ((-14.0, 28.0)))
	sim.setAgentPosition(9, ((20.0, 30.0)))
	sim.setAgentPosition(10, ((17.0, 30.0)))
	sim.setAgentPosition(11, ((14.0, 30.0)))
	sim.setAgentPosition(12, ((20.0, 29.0)))
	sim.setAgentPosition(13, ((17.0, 29.0)))
	sim.setAgentPosition(14, ((14.0, 29.0)))
	sim.setAgentPosition(15, ((20.0, 28.0)))
	sim.setAgentPosition(16, ((17.0, 28.0)))
	sim.setAgentPosition(17, ((14.0, 28.0)))
	'''
	sim.setAgentPosition(0, ((-20.0, 30.0)))
	sim.setAgentPosition(1, ((-17.0, 30.0)))
	sim.setAgentPosition(2, ((20.0, 30.0)))
	sim.setAgentPosition(3, ((17.0, 30.0)))

	for i in range(num_agents):
		if i < 2:
			sim.setAgentPrefVelocity(i, (1.0, 0.0))
		elif i > 1:
			sim.setAgentPrefVelocity(i, (-1.0, 0.0))

	goals = [0 for x in range(num_agents)]
	for x in range(num_agents):
		goals[x] = (-sim.getAgentPosition(x)[0], sim.getAgentPosition(x)[1])
		sim.setAgentGoal(x, (goals[x][0], goals[x][1]))
	sim.doStep()

	for agent in agents:
		agent = agent.agent_parameters
		own_positon = sim.getAgentPosition(agent)
		own_velocity = sim.getAgentVelocity(agent)
		relative_neighbor_positions = getNeighborPositions(agent)
		relative_neighbor_velocities = getRelativeNeighborVelocityList(agent)
		relative_obstacle_position = getDistanceToObstacle(agent)
		distance_to_goal = getDistanceToGoal(agent)
		
		nprnp = np.array(relative_neighbor_positions)
		nprnv = np.array(relative_neighbor_velocities)

		state[agent] = np.concatenate([own_positon, own_velocity, nprnp.flatten(), nprnv.flatten(), np.atleast_1d(relative_obstacle_position), distance_to_goal, np.atleast_1d(done)])

	return state

def getState(agent):
	#cambiar para solo un agente
	#state = [0 for x in range(num_agents)]
	done = False

	distance_to_goal = getDistanceToPoint(sim.getAgentPosition(a), sim.getAgentGoal(a))
	if distance_to_goal < 0.1:
		done = True
	#distance_to_goal = getDistanceToGoal(agent)
	
	#if distance_to_goal[0] < 0.1 and distance_to_goal[1] < 0.1:
	#	done = True
	#else:
	#	done = False
	own_positon = sim.getAgentPosition(agent)
	own_velocity = sim.getAgentVelocity(agent)
	relative_neighbor_positions = getNeighborPositions(agent)
	relative_neighbor_velocities = getRelativeNeighborVelocityList(agent)
	relative_obstacle_position = getDistanceToObstacle(agent)
	distance_to_goal = getDistanceToGoal(agent)

	nprnp = np.array(relative_neighbor_positions)
	nprnv = np.array(relative_neighbor_velocities)

	state[agent] = np.concatenate([own_positon, own_velocity, nprnp.flatten(), nprnv.flatten(), np.atleast_1d(relative_obstacle_position), distance_to_goal, np.atleast_1d(done)])

	return state

def render(poslist, a):
	pygame.init()
	screen_size = (display_width, display_height)
	gameDisplay = pygame.display.set_mode(screen_size, pygame.RESIZABLE, 32)
	pygame.display.set_caption('Tesis')

	black = (0,0,0)
	white = (255,255,255)
	red = (255,0,0)
	green = (0,255,0)
	orange = (255,69,0)

	clock = pygame.time.Clock()
	
	def agent(poslist, color):
		counter = 0
		for p in poslist:
			if counter == a:
				pygame.draw.circle(gameDisplay, red, center_origin((p[0],p[1])), 1.0)
				pygame.draw.circle(gameDisplay, orange, center_origin((p[2],p[3])), 1.0)
			else:
				pygame.draw.circle(gameDisplay, green, center_origin((p[0],p[1])), 1.0)
				pygame.draw.circle(gameDisplay, black, center_origin((p[2],p[3])), 1.0)
			counter += 1

	rect = 0.0, 100.0, 400.0, 150.0
	gameDisplay.fill(white)
	pygame.draw.line(gameDisplay, (0,0,255), center_origin((200.0, 31.0)), center_origin((-200.0, 31.0)), 1)
	pygame.draw.line(gameDisplay, (0,0,255), center_origin((200.0, 27.4)), center_origin((-200.0, 27.4)), 1)
	agent(poslist, green)

	pygame.display.update()
	clock.tick()

def center_origin(p):
    return (p[0] + display_width // 2, p[1] + display_height // 2)

#timestep, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed
sim = rvo2.PyRVOSimulator(time_step, NEIGHBOR_DIST, max_neighbors, time_horizon_obst, time_horizon_obst, radius, max_speed)  

# Pass either just the position (the other parameters then use
# the default values passed to the PyRVOSimulator constructor),
# or pass all available parameters.

#Initialize initial parameters

for i in range(num_agents):
	sim.setAgentDefaults(neighbor_distance, num_neighbors, time_horizon_ORCA , time_horizon_obst_ORCA , radius_ORCA , max_speed_ORCA)

#Network input: own_positon, own_velocity, relative_neighbor_positions, relative_neighbor_velocities, relative_obstacle_position, distance_to_goal, done
#Network output: Velocity

#NETWORK
class critic(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(128, activation='relu')
    self.v = tf.keras.layers.Dense(1, activation = None)

  def call(self, input_data):
    x = self.d1(input_data)
    v = self.v(x)
    return v
    
class actor(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(128,activation='relu')
    #Output layers. Sigmoid for translational velocity and tanh for rotational velocity.
    self.a = tf.keras.layers.Dense(1,activation='sigmoid')
    self.b = tf.keras.layers.Dense(1,activation='tanh')

  def call(self, input_data):
    x = self.d1(input_data)
    a = self.a(x)
    b = self.b(x)
    c = tf.concat([a,b], 1)
    return c

class agent():
	def __init__(self, gamma = 0.99, std = [-0.69, -0.69]):
		self.gamma = gamma
		self.std = std
		self.a_opt = tf.keras.optimizers.Adam(learning_rate=5e-5)
		self.c_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
		self.actor = actor()
		self.critic = critic()
		self.clip_pram = 0.2
		self.agent_parameters = 0
          
	def act(self, state):
		logstd = tf.Variable(tf.ones((1,2))*self.std, tf.float32)
		prob = self.actor(np.array([state]))
		std = tf.math.exp(logstd)
		#std = tf.expand_dims(aux, prob)
		#std = std.numpy()
		prob = prob.numpy()
		dist = tfp.distributions.Normal(prob, std)
		action = dist.sample()
		return action.numpy()[0]

	def actor_loss(self, probs, actions, adv, old_probs, closs):
		loss = 0
		probability = probs/old_probs
	
		kl = tf.keras.losses.KLDivergence()
		KL = kl(old_probs, probs).numpy()

		for t in range(len(rewards)):
			loss = loss + probability[t]*adv[t] - BETA*KL + XI*(max(0, KL-2*KLtarget)**2)

		return loss

	def cricit_loss(self, rewards, v):
		loss = 0
		for t in range(len(rewards)):
			aux = 0
			for tt in range(len(rewards)-1, t, -1):
				aux = rewards[tt]
				aux = (GAMMA**(tt-t))*rewards[tt]
			loss = loss + aux - v[t]**2

		return loss

	def learn(self, states, actions,  adv , old_probs, discnt_rewards):
		discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
		adv = tf.reshape(adv, (len(adv),))

		old_p = old_probs

		old_p = tf.reshape(old_p, (len(old_p),2))
		with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
			p = self.actor(states, training=True)
			v =  self.critic(states,training=True)
			v = tf.reshape(v, (len(v),))
			td = tf.math.subtract(discnt_rewards, v)
			c_loss = self.cricit_loss(rewards, v)
			a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)
            
		grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
		grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
		self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
		self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
		return a_loss, c_loss

def testReward(agent, a):
	print ("Entered test reward")
	total_reward = 0
	state = reset()
	print ("state reset ")
	done = False
	dones = []
	i = 0
	ts = 0
	while len(dones) != 4:
		all_aloss = []
		all_closs = []
		rewards = []
		states = []
		actions = []
		probs = []
		values = []

		print ("ts:", ts)
		action = agent.actor(np.array([state[a]])).numpy()[0]
		print ("action: ", action)
		#print("action: ", action)
		if action[0] > 1.0:
			action[0] = 1.0
		if action[1] > 1.0:
			action[1] = 1.0
		if action[1] < -1.0:
			action[1] = -1.0
			
		angle = action[1] * 180/math.pi
		pref_velocity_x = action[0] * math.cos(angle)
		pref_velocity_y = action[1] * math.sin(angle)
		pref_velocity = (pref_velocity_x, pref_velocity_y)
		sim.setAgentPrefVelocity(a, pref_velocity)
		old_agent_position = sim.getAgentPosition(a)
		sim.doStep()
		next_state = getState(a)
		sim.setAgentPrefVelocity(a, pref_velocity)
		new_agent_position = sim.getAgentPosition(a)
		goal = sim.getAgentGoal(a)
		goal_position_difference = math.sqrt((old_agent_position[0] - goal[0])**2 + (old_agent_position[1] - goal[1])**2) - math.sqrt((new_agent_position[0] - goal[0])**2 + (new_agent_position[1] - goal[1])**2) 
		print ("goal_position_difference: ", goal_position_difference)
		reward = getReward(a, goal_position_difference)
		done = next_state[a][len(next_state[a])-1]

		state = next_state
		total_reward += reward

		ts += 1
		print ("ts:", ts)
		if ts == 4000:
			break

	return total_reward

def gae(states, actions, rewards, values, gamma):
	a = 0
	lmbda = 0.95
	returns = []

	for t in range(len(rewards)):
		delta = rewards[t] + gamma * values[t + 1] - values[t]
		a = a + ((gamma * lmbda)**i) * delta
		returns.append(a)

	adv = np.array(returns, dtype=np.float32)
	states = np.array(states, dtype=np.float32)
	actions = np.array(actions, dtype=np.int32)
	returns = np.array(returns, dtype=np.float32)
	return states, actions, returns, adv    

#MAIN START
#Assign initial positions

tf.random.set_seed(336699)
'''
agent0 = agent()
agent0.agent_parameters = sim.addAgent((-20.0, 30.0))
a0 = agent0.agent_parameters
agent1 = agent()
agent1.agent_parameters = sim.addAgent((-17.0, 30.0))
a1 = agent1.agent_parameters
agent2 = agent()
agent2.agent_parameters = sim.addAgent((-14.0, 30.0))
a2 = agent2.agent_parameters
agent3 = agent()
agent3.agent_parameters = sim.addAgent((-20.0, 29.0))
a3 = agent3.agent_parameters
agent4 = agent()
agent4.agent_parameters = sim.addAgent((-17.0, 29.0))
a4 = agent4.agent_parameters
agent5 = agent()
agent5.agent_parameters = sim.addAgent((-14.0, 29.0))
a5 = agent5.agent_parameters
agent6 = agent()
agent6.agent_parameters = sim.addAgent((-20.0, 28.0))
a6 = agent6.agent_parameters
agent7 = agent()
agent7.agent_parameters = sim.addAgent((-17.0, 28.0))
a7 = agent7.agent_parameters
agent8 = agent()
agent8.agent_parameters = sim.addAgent((-14.0, 28.0))
a8 = agent8.agent_parameters
agent9 = agent()
agent9.agent_parameters = sim.addAgent((20.0, 30.0))
a9 = agent9.agent_parameters
agent10 = agent()
agent10.agent_parameters = sim.addAgent((17.0, 30.0))
a10 = agent10.agent_parameters
agent11 = agent()
agent11.agent_parameters = sim.addAgent((14.0, 30.0))
a11 = agent11.agent_parameters
agent12 = agent()
agent12.agent_parameters = sim.addAgent((20.0, 29.0))
a12 = agent12.agent_parameters
agent13 = agent()
agent13.agent_parameters = sim.addAgent((17.0, 29.0))
a13 = agent13.agent_parameters
agent14 = agent()
agent14.agent_parameters = sim.addAgent((14.0, 29.0))
a14 = agent14.agent_parameters
agent15 = agent()
agent15.agent_parameters = sim.addAgent((20.0, 28.0))
a15 = agent15.agent_parameters
agent16 = agent()
agent16.agent_parameters = sim.addAgent((17.0, 28.0))
a16 = agent16.agent_parameters
agent17 = agent()
agent17.agent_parameters = sim.addAgent((14.0, 28.0))
a17 = agent17.agent_parameters
'''
agent0 = agent()
agent0.agent_parameters = sim.addAgent((-20.0, 30.0))
a0 = agent0.agent_parameters
agent1 = agent()
agent1.agent_parameters = sim.addAgent((-17.0, 30.0))
a1 = agent1.agent_parameters
agent2 = agent()
agent2.agent_parameters = sim.addAgent((20.0, 30.0))
a2 = agent2.agent_parameters
agent3 = agent()
agent3.agent_parameters = sim.addAgent((17.0, 30.0))
a3 = agent3.agent_parameters

#agents = (agent0, agent1, agent2, agent3, agent4, agent5, agent6, agent7, agent8, agent9, agent10, agent11, agent12, agent13, agent14, agent15, agent16, agent17)
agents = (agent0, agent1, agent2, agent3)

#agent_list = (a0, a1, a2, a3, a4, a5, a6, a7, a8, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17)
agent_list = (a0, a1, a2, a3)

# Declare obstacles
obstacle1 = [(200.0, 31.0), (-200.0, 31.0), (-200.0, 30.6), (200.0, 30.6)]
obstacle2 = [(200.0, 27.4), (-200.0, 27.4), (-200.0, 27.0), (200.0, 27.0)]
obstacle1_y = 30.6
obstacle2_y = 27.4

o1 = sim.addObstacle([(200.0, 31.0), (-200.0, 31.0), (-200.0, 30.6), (200.0, 30.6)])
o2 = sim.addObstacle([(200.0, 27.4), (-200.0, 27.4), (-200.0, 27.0), (200.0, 27.0)])

sim.processObstacles()

#Assign Goals
goals = [0 for x in range(num_agents)]
for x in range(num_agents):
	goals[x] = (-sim.getAgentPosition(x)[0], sim.getAgentPosition(x)[1])
	sim.setAgentGoal(x, (goals[x][0], goals[x][1]))

print('Simulation has %i agents and %i obstacle vertices in it.' %
      (sim.getNumAgents(), sim.getNumObstacleVertices()))

print('Running simulation')

steps = 50
ep_reward = []
total_avgr = []
best_reward = 0
rewards_list = []
white = (0 ,0, 0)
green = (0, 255, 0)

for s in range(steps): 

	done = False
	state = reset()
	all_aloss = []
	all_closs = []
	rewards = []
	states = []
	actions = []
	probs = []
	dones = []
	values = []
	count = 0
	
	print("New Step", s)

	for agent in agents:
		i = 0
		ts = 0
		while len(dones) != 4:
			a = agent.agent_parameters
			print ("========== AGENT", a, " STEP", s, "==========")
			all_aloss = []
			all_closs = []
			rewards = []
			states = []
			actions = []
			probs = []
			values = []
			
			action = agent.act(state[a])

			if action[0] > 1.0:
				action[0] = 1.0
			if action[0] < 0.0:
				action[0] = 0.0
			if action[1] > 1.0:
				action[1] = 1.0
			if action[1] < -1.0:
				action[1] = -1.0
			
			angle = action[1] * 180/math.pi
			pref_velocity_x = action[0] * math.cos(angle)
			pref_velocity_y = action[1] * math.sin(angle)
			value = agent.critic(np.array([state[a]])).numpy()
			pref_velocity = (pref_velocity_x, pref_velocity_y)
			sim.setAgentPrefVelocity(a, pref_velocity)
			print ("action: ", action)
			old_agent_position = sim.getAgentPosition(a)
			sim.doStep()
			sim.setAgentPrefVelocity(a, pref_velocity)
			next_state = getState(a)
			new_agent_position = sim.getAgentPosition(a)
			goal = sim.getAgentGoal(a)
			goal_position_difference = math.sqrt((old_agent_position[0] - goal[0])**2 + (old_agent_position[1] - goal[1])**2) - math.sqrt((new_agent_position[0] - goal[0])**2 + (new_agent_position[1] - goal[1])**2) 
			reward = getReward(a, goal_position_difference)
			done = state[a][len(state[a])-1]
			rewards.append(reward)
			states.append(state[a])
			actions.append(action)
			prob = agent.actor(np.array([state[a]]))
			probs.append(prob[0])
			values.append(value[0][0])
			state = next_state
			
			print ("position: ", sim.getAgentPosition(a))
			
			poslist = []
			goallist = []
			for b in agent_list:
				posx = sim.getAgentPosition(b)[0]
				posy = sim.getAgentPosition(b)[1] 
				goalx = sim.getAgentGoal(b)[0]
				goaly = sim.getAgentGoal(b)[1]
				pos = (posx, posy, goalx, goaly)
				poslist.append(pos)
			render(poslist, a)
			
			ts += 1 
			#print ("ts: ", ts)
			if done or ts == 4000:
				print ("Entered if done ")
				dones.append(1-done)
				state = reset()
				break
			i += 1

			value = agent.critic(np.array([state[a]])).numpy()
			values.append(value[0][0])
			np.reshape(probs, (len(probs), 2))
			probs = np.stack(probs, axis=0)

			states, actions, returns, adv  = gae(states, actions, rewards, values, 1)
		'''
		count += i
		if count > Tmax:
			print ("Count = ", count)
			break
		'''
	print ("============ END DATA COLLECTION ==============")

	old_probs = probs

	for epocs in range(10):
		al,cl = agent.learn(states, actions, adv, probs, returns)
		#Adapt KL penalty coefficient
		p = agent.actor(states, training=True)
		kl1 = tf.keras.losses.KLDivergence()
		KL1 = kl1(old_probs, p).numpy()
		print ("Finished Learn for epoc ", epocs)
		#Si se calculan las recompensas fueras del for explota
		reward = testReward(agent, a)
		print(f"total test reward is {reward}")
		rewards_list.append(reward)
		reset()

	if (KL1 > BETAhigh*KLtarget):
		BETA = ALPHA*BETA
	elif (KL1 < BETAlow*KLtarget):
		BETA = BETA/ALPHA
			
print("========== END STEP ", s, "==========")

print (rewards_list)

#TODO

'''
1) End for if AllDone. DONE, need to test
2) Check the objective of the reward function.
'''
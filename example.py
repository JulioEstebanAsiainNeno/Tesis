
#!/usr/bin/env python

import rvo2
import pygame
import time
import os.path

sim = rvo2.PyRVOSimulator(1/60., 1.5, 5, 1.5, 2, 0.4, 1.5)

# Pass either just the position (the other parameters then use
# the default values passed to the PyRVOSimulator constructor),
# or pass all available parameters.
a0 = sim.addAgent((-20.0, 30.0)) #(-20.0 ,30.0), (-27.2, 39.0). Old diff = 17.0 + 2r + 2.2 = 20.0. New diff = 17.0 + 2r + 2.2 = 27.2
a1 = sim.addAgent((-17.0, 30.0)) 
a2 = sim.addAgent((-14.0, 30.0)) 
a3 = sim.addAgent((-20.0, 29.0)) 
a4 = sim.addAgent((-17.0, 29.0)) #(-17.0, 29.0), (-17.0, 29.0)  
a5 = sim.addAgent((-14.0, 29.0)) 
a6 = sim.addAgent((-20.0, 28.0))
a7 = sim.addAgent((-17.0, 28.0))
a8 = sim.addAgent((-14.0, 28.0)) #(-14.0, 28.0), (-6.8, 19.0)  
a9 = sim.addAgent((20.0, 30.0))
a10 = sim.addAgent((17.0, 30.0))
a11 = sim.addAgent((14.0, 30.0))
a12 = sim.addAgent((20.0, 29.0))
a13 = sim.addAgent((17.0, 29.0))
a14 = sim.addAgent((14.0, 29.0))
a15 = sim.addAgent((20.0, 28.0))
a16 = sim.addAgent((17.0, 28.0))
a17 = sim.addAgent((14.0, 28.0))

agent_list = (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17)

# Obstacles are also supported.
o1 = sim.addObstacle([(200.0, 31.0), (-200.0, 31.0), (-200.0, 30.6), (200.0, 30.6)]) #31.0 Old diff = 30.0 + 1.5r
o2 = sim.addObstacle([(200.0, 27.4), (-200.0, 27.4), (-200.0, 27.0), (200.0, 27.0)]) #27.4 New diff = 39.0 + 1.5r
sim.processObstacles()

sim.setAgentPrefVelocity(a0, (1.5, 0.0))
sim.setAgentPrefVelocity(a1, (1.5, 0.0))
sim.setAgentPrefVelocity(a2, (1.5, 0.0))
sim.setAgentPrefVelocity(a3, (1.5, 0.0))
sim.setAgentPrefVelocity(a4, (1.5, 0.0))
sim.setAgentPrefVelocity(a5, (1.5, 0.0))
sim.setAgentPrefVelocity(a6, (1.5, 0.0))
sim.setAgentPrefVelocity(a7, (1.5, 0.0))
sim.setAgentPrefVelocity(a8, (1.5, 0.0))
sim.setAgentPrefVelocity(a9, (-1.5, 0.0))
sim.setAgentPrefVelocity(a10, (-1.5, 0.0))
sim.setAgentPrefVelocity(a11, (-1.5, 0.0))
sim.setAgentPrefVelocity(a12, (-1.5, 0.0))
sim.setAgentPrefVelocity(a13, (-1.5, 0.0))
sim.setAgentPrefVelocity(a14, (-1.5, 0.0))
sim.setAgentPrefVelocity(a15, (-1.5, 0.0))
sim.setAgentPrefVelocity(a16, (-1.5, 0.0))
sim.setAgentPrefVelocity(a17, (-1.5, 0.0))

display_width = 400
display_height = 300

background = pygame.image.load(os.path.join('blank.png'))
SCREEN_WIDTH = background.get_rect().size[0]
SCREEN_HEIGHT = background.get_rect().size[1]


clock = pygame.time.Clock()
zoom_event = False
scale_up = 1.2
scale_down = 0.8

class GameState:
    def __init__(self):
        self.tab = 1
        self.zoom = 1
        self.world_offset_x = 0
        self.world_offset_y = 0
        self.update_screen = True
        self.panning = False
        self.pan_start_pos = None
        self.legacy_screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

game_state = GameState()

def world_2_screen(world_x, world_y):
    screen_x = (world_x - game_state.world_offset_x) * game_state.zoom
    screen_y = (world_y - game_state.world_offset_y) * game_state.zoom
    return [screen_x, screen_y]


def screen_2_world(screen_x, screen_y):
    world_x = (screen_x / game_state.zoom) + game_state.world_offset_x
    world_y = (screen_y / game_state.zoom) + game_state.world_offset_y
    return [world_x, world_y]


def render(poslist):

	#gameDisplay = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
	background = pygame.image.load(os.path.join('blank.png'))
	screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
	pygame.display.set_caption('Tesis')

	black = (0,0,0)
	white = (255,255,255)

	clock = pygame.time.Clock()
	
	def car(poslist):
		for p in poslist:
			circle = pygame.draw.circle(background, (0,255,0), center_origin((p[0],p[1])), 1.0)
			#screen.blit(screen, circle)

	rect = pygame.draw.rect(background, (0,255,0), (200.0, 100.0, 400.0, 150.0))

	#gameDisplay.fill(white)
	line1 = pygame.draw.line(background, (0,0,255), center_origin((200.0, 31.0)), center_origin((-200.0, 31.0)), 1)
	line2 = pygame.draw.line(background, (0,0,255), center_origin((200.0, 27.4)), center_origin((-200.0, 27.4)), 1)

	car(poslist)

	pygame.display.update()
	clock.tick(30)

	# Mouse screen coords
	mouse_x, mouse_y = pygame.mouse.get_pos()

	# event handler
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_RETURN:
				if game_state.tab == 1:
					game_state.tab = 2
				elif game_state.tab == 2:
					game_state.tab = 1
			elif event.key == pygame.K_p:
				pygame.image.save(screen, "NEW.png")

		elif event.type == pygame.MOUSEBUTTONDOWN:
			if event.button == 4 or event.button == 5:
				# X and Y before the zoom
				mouseworld_x_before, mouseworld_y_before = screen_2_world(mouse_x, mouse_y)

                # ZOOM IN/OUT
				if event.button == 4 and game_state.zoom < 10:
					game_state.zoom *= scale_up
				elif event.button == 5 and game_state.zoom > 0.5:
					game_state.zoom *= scale_down

				# X and Y after the zoom
				mouseworld_x_after, mouseworld_y_after = screen_2_world(mouse_x, mouse_y)

				# Do the difference between before and after, and add it to the offset
				game_state.world_offset_x += mouseworld_x_before - mouseworld_x_after
				game_state.world_offset_y += mouseworld_y_before - mouseworld_y_after

			elif event.button == 1:
				# PAN START
				game_state.panning = True
				game_state.pan_start_pos = mouse_x, mouse_y

		elif event.type == pygame.MOUSEBUTTONUP:
			if event.button == 1 and game_state.panning:
				# PAN STOP
				game_state.panning = False

	if game_state.panning:
		# Pans the screen if the left mouse button is held
		game_state.world_offset_x -= (mouse_x - game_state.pan_start_pos[0]) / game_state.zoom
		game_state.world_offset_y -= (mouse_y - game_state.pan_start_pos[1]) / game_state.zoom
		game_state.pan_start_pos = mouse_x, mouse_y

	# Draw the screen
	if game_state.tab == 1:
		if game_state.update_screen:
			# Updates the legacy screen if something has changed in the image data
			game_state.legacy_screen.blit(background, (0, 0))
			game_state.update_screen = False

		# Sets variables for the section of the legacy screen to be zoomed
		world_left, world_top = screen_2_world(0, 0)
		world_right, world_bottom = SCREEN_WIDTH/game_state.zoom, SCREEN_HEIGHT/game_state.zoom

		# Makes a temp surface with the dimensions of a smaller section of the legacy screen (for zooming).
		new_screen = pygame.Surface((world_right, world_bottom))
		# Blits the smaller section of the legacy screen to the temp screen
		new_screen.blit(game_state.legacy_screen, (0, 0), (world_left, world_top, world_right, world_bottom))
		# Blits the final cut-out to the main screen, and scales the image to fit with the screen height and width
		screen.fill((255, 255, 255))
		screen.blit(pygame.transform.scale(new_screen, (SCREEN_WIDTH, SCREEN_HEIGHT)), (0, 0))
	
	#car(poslist)

	# looping


def center_origin(p):
    return (p[0] + SCREEN_WIDTH // 2, p[1] + SCREEN_HEIGHT // 2)

print('Simulation has %i agents and %i obstacle vertices in it.' %
      (sim.getNumAgents(), sim.getNumObstacleVertices()))

print('Running simulation')

for step in range(1500):
	sim.doStep()
	poslist = []

	for a in agent_list:
		posx = sim.getAgentPosition(a)[0]
		posy = sim.getAgentPosition(a)[1] 
		'''
		offset_x = 7.2
		offset_y = 10.0
		offset_d = 10.0
		if a == 0 or a == 1 or a == 2 or a == 9 or a == 10 or a == 11:
			posy = sim.getAgentPosition(a)[1] + offset_y
		if a == 6 or a == 7 or a == 8 or a == 15 or a == 16 or a == 17:
			posy = sim.getAgentPosition(a)[1] - offset_y
		if a == 0 or a == 3 or a == 6:
			posx = sim.getAgentPosition(a)[0] - offset_x
		if a == 2 or a == 5 or a == 8:
			posx = sim.getAgentPosition(a)[0] + offset_x 
		if a == 9 or a == 12 or a == 15:
			posx = sim.getAgentPosition(a)[0] + offset_x 
		if a == 11 or a == 14 or a == 17:
			posx = sim.getAgentPosition(a)[0] - offset_x
		'''
		pos = (posx, posy)
		poslist.append(pos)
	render(poslist)
	time.sleep(0.01)

	positions = ['(%5.3f, %5.3f)' % sim.getAgentPosition(agent_no) for agent_no in (a0, a1, a2, a3)]
	velocities = ['(%5.3f, %5.3f)' % sim.getAgentVelocity(agent_no) for agent_no in (a0, a1, a2, a3)]
	print('step=%2i t=%.3f  p=%s v=%s' % (step, sim.getGlobalTime(), '  '.join(positions), ' '.join(velocities)))

pygame.quit()


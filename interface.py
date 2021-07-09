import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

def render(pos):
	pygame.init()

	display_width = 800
	display_height = 600

	gameDisplay = pygame.display.set_mode((display_width, display_height))
	pygame.display.set_caption('Tesis')

	surf = pygame.Surface((display_width/2, display_height/2))
	gameDisplay.blit(surf, (0,0))

	pygame.transform.scale(surf, ((display_width//200)*150, (display_height//200)*150))

	black = (0,0,0)
	white = (255,255,255)

	clock = pygame.time.Clock()
	crashed = False
	agent_image = pygame.image.load('agent21.png')

	def car(x,y):
		agent = pygame.draw.circle(gameDisplay, (0,255,0), (x,y), 10)

	x = (display_width * 0.45)
	y = (display_height * 0.8)
	x_change = 0
	car_speed = 0
	i = 1

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			crashed = True

	x += pos
	i = i+1
	gameDisplay.fill(white)
	glTranslatef(0.0, 0.0, -5.0)
	car(x,y)
        
	pygame.display.update()
	pygame.time.wait(100)
	clock.tick(60)

pos = 0
for a in range(125):
	render(pos)
	pos = pos + 1
	print (pos)

pygame.quit()
#quit()
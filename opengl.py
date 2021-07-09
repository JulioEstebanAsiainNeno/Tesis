import pygame
import OpenGL

from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from math import *

def drawCircle():
    posx, posy = 2,2
    sides = 32
    radius = 0.4
    glBegin(GL_POLYGON)
    for i in range(50):
        cosine = radius * cos(i*2*pi/sides) + posx
        sine = radius * sin(i*2*pi/sides) + posy
        glVertex2f(cosine,sine)
    glEnd()

def main():
    pygame.init()

    display_width = 800
    display_height = 600

    pygame.display.set_mode((display_width, display_height), DOUBLEBUF|OPENGL)

    gluPerspective(45.0, (display_width/display_height), 1, 50.0)
    glTranslatef(0.0, 0.0, 0.0)
    glRotatef(20, 0, 0, 0)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        drawCircle()
        pygame.display.flip()
        pygame.time.wait(10)

main()
import sys
import pygame as pg
import numpy as np
import math
from OpenGL.GL import *
from debug import debug_gl, debug_callback
from OpenGL.GL.shaders import compileProgram, compileShader
from vertex_array import VertexArray
from vertex_buffer import VertexBuffer
from vertex_buffer_layout import VertexBufferLayout
from shader import Shader
from renderer import Renderer

class App:
    def __init__(self) -> None:
        #init pygame
        pg.init()
        pg.display.set_mode((500, 500), pg.OPENGL | pg.DOUBLEBUF)
        self.clock = pg.time.Clock()
        #init openGL
        # print(debug_gl(glGetString)(GL_VERSIONE))
        print(glGetString(GL_VERSION))
        print(glGetString(GL_SHADING_LANGUAGE_VERSION))
        # print("Max vertex attribute", glGetIntegerv(GL_MAX_VERTEX_ATTRIBS))
        # glEnable(GL_DEBUG_OUTPUT)
        # glDebugMessageCallback(debug_callback, 0)

    def createShaderAuto(self, vertexFilePath, fragmentFilePath):
        with open(vertexFilePath, "r") as f:
            vertex_src = f.readlines()

        with open(fragmentFilePath, "r") as f:
            fragment_src = f.readlines()

        shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER)
        )

        return shader

    def compileShader(self, type, source):
        id = glCreateShader(type)
        glShaderSource(id, source)
        glCompileShader(id)

        result = glGetShaderiv(id, GL_COMPILE_STATUS)
        assert result, "Compile shader failed"
        return id

    def createShaderNormal(
            self, 
            vertexShaderPath, 
            fragmentShaderPath):
        with open(vertexShaderPath, "r") as vertex_file:
            vertex_src = "".join(vertex_file.readlines())
        with open(fragmentShaderPath, "r") as fragment_file:
            fragment_src = "".join(fragment_file.readlines())
        # print(vertex_src)
        # print(fragment_src)
        program = glCreateProgram()
        vs = self.compileShader(GL_VERTEX_SHADER, vertex_src)
        fs = self.compileShader(GL_FRAGMENT_SHADER, fragment_src)
        glAttachShader(program, vs)
        glAttachShader(program, fs)
        glLinkProgram(program)
        glValidateProgram(program)

        glDeleteShader(vs)
        glDeleteShader(fs)

        return program

    def load_image(self, path):
        return pg.image.load(path)

    def image_to_bytes(self, image: pg.Surface, format="RGB"):
        return pg.image.tostring(image, format)

    def mainloop(self):
        #setup texture
        image = self.load_image("./textures/container.jpg")
        # image = self.load_image("./textures/wall.jpg")
        image1 = pg.transform.flip(self.load_image("./textures/awesomeface.png"), flip_x=False, flip_y=True)
        glActiveTexture(GL_TEXTURE0)
        texture_id = glGenTextures(2)
        
        glBindTexture(GL_TEXTURE_2D, texture_id[0])
        glGenerateMipmap(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB, 
            image.get_width(),
            image.get_height(),
            0, GL_RGB, GL_UNSIGNED_BYTE, self.image_to_bytes(image))

        glActiveTexture(GL_TEXTURE0 + 1)
        glBindTexture(GL_TEXTURE_2D, texture_id[1])
        glGenerateMipmap(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, 
            image1.get_width(),
            image1.get_height(),
            0, GL_RGBA, GL_UNSIGNED_BYTE, self.image_to_bytes(image1, "RGBA"))


        self.shader = Shader(
            # "./shaders/triangle.vert",
            # "./shaders/triangle.frag"
            "./shaders/rect.vert",
            "./shaders/rect.frag"
        )
        self.renderer = Renderer()
        self.rectangle = Rectangle()
        mix_scale = 0
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False

            self.renderer.clear()
            # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            self.shader.setInt("outTexture", 0)
            self.shader.setInt("outTexture1", 1)
            # print(mix_scale)
            self.shader.setFloat("mixScale", mix_scale)
            # debug_gl(self.shader.setFloat)("u_deltaTime", math.sin(dt) / 2 + 0.5)
            # self.shader.setFloat("offsetPosition", position)
            self.renderer.draw(
                self.rectangle.va,
                self.rectangle.eb,
                self.shader
            )

            mix_scale += 0.002
            if mix_scale > 1:
                mix_scale = 0

            pg.display.flip()
            self.clock.tick(60)
        self.quit()

    def quit(self):
        pg.quit()
        sys.exit()

class Rectangle:
    def __init__(self) -> None:
        # self.position = [
        #     #position       color      texture
        #      0.5,  0.5,    1, 0, 0,     2,  2,
        #      0.5, -0.5,    0, 1, 0,     2,  0,
        #     -0.5, -0.5,    0, 0, 1,     0,  0,
        #     -0.5,  0.5,    1, 0, 0,     0,  2
        # ]

        self.position = [
            #position       color      texture
             1,  1,    1, 0, 0,     2,  2,
             1, -1,    0, 1, 0,     2,  0,
            -1, -1,    0, 0, 1,     0,  0,
            -1,  1,    1, 0, 0,     0,  2
        ]

        self.indices = [
            0, 1, 2,
            0, 2, 3
        ]

        self.va = VertexArray()
        self.vb = VertexBuffer(self.position, VertexBuffer.ARRAY_BUFFER)
        self.layout_vb = VertexBufferLayout()
        self.layout_vb.add_element(VertexBufferLayout.FLOAT, 2)
        self.layout_vb.add_element(VertexBufferLayout.FLOAT, 3)
        self.layout_vb.add_element(VertexBufferLayout.FLOAT, 2)

        self.va.add_buffer(self.vb, self.layout_vb)

        self.eb = VertexBuffer(self.indices, VertexBuffer.ELEMENT_ARRAY_BUFFER, data_type=VertexBuffer.UNSIGNED_INT_32)

        self.va.unbind()

if __name__ == "__main__":
    myApp = App()
    myApp.mainloop()

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
        print("Max vertex attribute", glGetIntegerv(GL_MAX_VERTEX_ATTRIBS))
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

    def mainloop(self):
        self.shader = Shader(
            "./shaders/triangle.vert",
            "./shaders/triangle.frag"
        )
        self.renderer = Renderer()
        self.triangle = Triangle()
        # self.triangle2 = Triangle()
        running = True
        dt = 0
        position = -2
        import asyncio
        asyncio.gather()
        # glUseProgram(self.shader)
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False

            self.renderer.clear()
            # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            self.shader.setFloat("u_deltaTime", math.sin(dt) / 2 + 0.5)
            # debug_gl(self.shader.setFloat)("u_deltaTime", math.sin(dt) / 2 + 0.5)
            # self.shader.setFloat("offsetPosition", position)
            self.renderer.draw(
                self.triangle.va,
                self.triangle.eb,
                self.shader
            )

            position += 0.003
            dt += 0.01
            if dt > 1:
                dt = 0

            pg.display.flip()
            self.clock.tick(60)
        self.quit()

    def quit(self):
        pg.quit()
        sys.exit()

class Triangle:
    def __init__(self) -> None:
        #values position
        # self.position = [
        #      0.5,  0.5,
        #      0.5, -0.5,
        #     -0.5, -0.5
        # ]

        self.position = [
               0,  0.5, 1, 0, 0, 
             0.5, -0.5, 0, 1, 0,
            -0.5, -0.5, 0, 0, 1,
            #  0.5, -0.5, 1, 0, 0
        ]

        # self.indices = [
        #     0, 1, 2
        # ]

        self.indices = [
            0, 1, 2,
            # 0, 2, 3
        ]

        # self.indices = np.array(self.indices, dtype=np.uint32)

        self.va = VertexArray()
        self.vb = VertexBuffer(self.position, VertexBuffer.ARRAY_BUFFER)
        self.layout_vb = VertexBufferLayout()
        self.layout_vb.add_element(VertexBufferLayout.FLOAT, 2)
        self.layout_vb.add_element(VertexBufferLayout.FLOAT, 3)

        # ebo = glGenBuffers(1)
        # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        # glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        self.va.add_buffer(self.vb, self.layout_vb)
        self.eb = VertexBuffer(self.indices, VertexBuffer.ELEMENT_ARRAY_BUFFER, data_type=VertexBuffer.UNSIGNED_INT_32)

        # ebo = glGenBuffers(1)
        # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        # glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        self.va.unbind()

if __name__ == "__main__":
    myApp = App()
    myApp.mainloop()

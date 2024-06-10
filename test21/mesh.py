from typing import List
from ctypes import c_void_p
import numpy as np
from numpy import array
from OpenGL.GL import *
from shader import Shader

class Vertex:
    def __init__(self, position, normal, tex_coords) -> None:
        self.position = position
        self.normal = normal
        self.tex_coords = tex_coords

class Texture:
    def __init__(self, id, type, path) -> None:
        self.id = id
        self.type = type
        self.path = path

class Mesh:
    def __init__(self, vertices: List[Vertex], indices: List[int], textures: List[Texture]) -> None:
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.vertices = vertices
        self.indices = indices
        self.textures = textures

        self.setup_mesh()

    def setup_mesh(self):
        self.vao = glGenVertexArrays(1)
        self.vbo, self.ebo = glGenBuffers(2)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        vertices = []
        [vertices.extend([*vertex.position, *vertex.normal, *vertex.tex_coords]) for vertex in self.vertices]
        vertices = array(vertices, dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        indices = array(self.indices, dtype=np.uint32)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        size = np.float32().itemsize
        
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * size, c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * size, c_void_p(3 * size))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * size, c_void_p(6 * size))

        glBindVertexArray(0)

    def draw(self, shader: Shader):
        diffuse_nr = 1
        specular_nr = 1
        for index, texture in enumerate(self.textures):
            glActiveTexture(GL_TEXTURE0 + index)
            glBindTexture(GL_TEXTURE_2D, texture.id)
            if texture.type == "texture_diffuse":
                shader.setInt(f"material.texture_diffuse{diffuse_nr}", index)
                # print(f"material.texture_diffuse{diffuse_nr}")
                diffuse_nr += 1
            elif texture.type == "texture_specular":
                shader.setInt(f"material.texture_specular{specular_nr}", index)
                # print(f"material.texture_specular{specular_nr}")
                specular_nr += 1
        
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
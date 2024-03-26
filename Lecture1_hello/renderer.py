from OpenGL.GL import *
from vertex_array import VertexArray
from vertex_buffer import VertexBuffer
from shader import Shader

class Renderer:
    def __init__(self) -> None:
        pass

    def draw(self, 
            vertex_array: VertexArray, 
            vertex_buffer: VertexBuffer, 
            shader: Shader):
        vertex_array.bind()
        vertex_buffer.bind()
        shader.use()
        glDrawElements(GL_TRIANGLES, vertex_buffer.data_size(), GL_UNSIGNED_INT, None)

    def clear(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
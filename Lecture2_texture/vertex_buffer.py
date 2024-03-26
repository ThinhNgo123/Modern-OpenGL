from OpenGL.GL import *
import numpy as np

class VertexBuffer:
    ARRAY_BUFFER = GL_ARRAY_BUFFER
    ELEMENT_ARRAY_BUFFER = GL_ELEMENT_ARRAY_BUFFER
    FLOAT32 = np.float32
    UNSIGNED_INT_32 = np.uint32
    def __init__(self, data, vertex_buffer_type, count=1, data_type=FLOAT32) -> None:
        self.data: np.array = np.array(data, dtype=data_type)
        self.type = vertex_buffer_type
        self.vbo = glGenBuffers(count)
        glBindBuffer(vertex_buffer_type, self.vbo)
        glBufferData(vertex_buffer_type, self.data.nbytes, self.data, GL_STATIC_DRAW)

    def bind(self):
        glBindBuffer(self.type, self.vbo)

    def unbind(self):
        glBindBuffer(self.type, 0)

    def data_size(self):
        return len(self.data)

    def nbytes(self):
        return self.data.nbytes

    def __delete__(self):
        glDeleteBuffers(1, [self.vbo])
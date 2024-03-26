from OpenGL.GL import *
from vertex_buffer import VertexBuffer
from vertex_buffer_layout import VertexBufferLayout
import ctypes

class VertexArray:
    def __init__(self, count=1) -> None:
        self.vao = glGenVertexArrays(count)

    def add_buffer(self, buffer: VertexBuffer, layout: VertexBufferLayout):
        self.bind()
        buffer.bind()
        offset = 0
        for index in range(len(layout.get_elements())):
            element = layout.get_element(index)
            glEnableVertexAttribArray(index)
            glVertexAttribPointer(
                index, 
                element.count, 
                element.type,
                element.normalized,
                layout.get_stride(),
                ctypes.c_void_p(offset))
            # print(layout.get_stride(), offset)
            offset += element.get_stride()
        # buffer.unbind()

    def bind(self):
        glBindVertexArray(self.vao)

    def unbind(self):
        glBindVertexArray(0)

    def __delete__(self):
        glDeleteVertexArrays(1, [self.vao])
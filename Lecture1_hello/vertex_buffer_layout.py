from OpenGL.GL import *

class VertexBufferElement:
    UNSIGNED_INT = GL_UNSIGNED_INT
    UNSIGNED_BYTE = GL_UNSIGNED_BYTE
    FLOAT = GL_FLOAT

    def __init__(self, type, count, normalized=GL_FALSE) -> None:
        self.type = type
        self.count = count
        self.normalized = normalized

    def get_stride(self):
        if self.type == VertexBufferElement.FLOAT or \
            self.type == VertexBufferElement.UNSIGNED_INT:
            return self.count * 4 # float and unsigned int is 4 byte
        elif self.type == VertexBufferElement.UNSIGNED_BYTE:
            return self.count # unsigned byte is 1 byte

class VertexBufferLayout:
    UNSIGNED_INT = GL_UNSIGNED_INT
    UNSIGNED_BYTE = GL_UNSIGNED_BYTE
    FLOAT = GL_FLOAT

    def __init__(self) -> None:
        self.elements: VertexBufferElement = []
        self.stride = 0

    def add_element(self, type, count):
        self.elements.append(VertexBufferElement(type, count))
        self.stride += self.elements[-1].get_stride()

    def get_elements(self):
        return self.elements

    def get_element(self, index) -> VertexBufferElement:
        return self.elements[index]

    def get_stride(self):
        return self.stride
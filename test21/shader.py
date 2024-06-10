from OpenGL.GL import *

class Shader:
    VERTEX_SHADER = GL_VERTEX_SHADER
    FRAGMENT_SHADER = GL_FRAGMENT_SHADER

    def __init__(self, vertex_shader_path, fragment_shader_path) -> None:
        vertex_shader_source = self.read_file(vertex_shader_path)
        fragment_shader_source = self.read_file(fragment_shader_path)
        vertex_id = self.compile_shader(Shader.VERTEX_SHADER, vertex_shader_source)
        fragment_id = self.compile_shader(Shader.FRAGMENT_SHADER, fragment_shader_source)
        self.program_id = self.compile_program(vertex_id, fragment_id)

        self.shader_location = {}

    def read_file(self, path):
        try:
            with open(path, "r") as file:
                source = file.read()
                return source
        except Exception as e:
            print(e)

    def compile_shader(self, type, source):
        try:
            shader_id = glCreateShader(type)
            glShaderSource(shader_id, source)
            glCompileShader(shader_id)
            status = glGetShaderiv(shader_id, GL_COMPILE_STATUS)
            if status:
                return shader_id
            raise Exception(glGetShaderInfoLog(shader_id))
        except Exception as e:
            print(e)

    def compile_program(self, vertex_shader_id, fragment_shader_id):
        try:
            program_id = glCreateProgram()
            glAttachShader(program_id, vertex_shader_id)
            glAttachShader(program_id, fragment_shader_id)
            glLinkProgram(program_id)
            glDeleteShader(vertex_shader_id)
            glDeleteShader(fragment_shader_id)
            status = glGetProgramiv(program_id, GL_LINK_STATUS)
            if status:
                return program_id
            raise Exception(glGetProgramInfoLog(program_id))
        except Exception as e:
            print(e)

    def use(self):
        glUseProgram(self.program_id)

    def setInt(self, name: str, value: int):
        glUniform1i(self.get_location_shader(name), value)
    
    def setBool(self, name: str, value: bool):
        glUniform1i(self.get_location_shader(name), int(value))

    def setFloat(self, name: str, value: float):
        # self.use()
        # location = self.get_location_shader(name)
        # print("location:", location)
        # print("error:", glGetError())
        # glUniform1f(location, value)
        # print("error:", glGetError())
        glUniform1f(self.get_location_shader(name), value)

    def setFloat3(self, name: str, value1: float, value2: float, value3: float):
        glUniform3f(self.get_location_shader(name), value1, value2, value3)

    def setMatrix4(self, name: str, matrix):
        glUniformMatrix4fv(self.get_location_shader(name), 1, GL_TRUE, matrix)

    def get_location_shader(self, name: str):
        location = self.shader_location.get(name, None)
        if location != None:
            return location
        # print("no location")
        location = glGetUniformLocation(self.program_id, name)
        self.shader_location[name] = location
        return location

    def __delete__(self):
        glDeleteProgram(self.program_id)
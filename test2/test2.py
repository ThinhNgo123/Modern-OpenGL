import pygame, sys, math
from pygame.locals import *
from OpenGL.GL import *
import numpy as np
import ctypes 

def translate(x, y, z):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [x, y, z, 1]
    ])

def scale(x, y, z):
    return np.array([
        [x, 0, 0, 0],
        [0, y, 0, 0],
        [0, 0, z, 0],
        [0, 0, 0, 1]
    ])

def rotate_x(a):
    return np.array([
        [1, 0, 0, 0],
        [0, math.cos(a), math.sin(a), 0],
        [0, -math.sin(a), math.cos(a), 0],
        [0, 0, 0, 1]
    ])


def rotate_y(a):
    return np.array([
        [math.cos(a), 0, -math.sin(a), 0],
        [0, 1, 0, 0],
        [math.sin(a), 0, math.cos(a), 0],
        [0, 0, 0, 1]
    ])


def rotate_z(a):
    return np.array([
        [math.cos(a), math.sin(a), 0, 0],
        [-math.sin(a), math.cos(a), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def read_file(path):
    try:
        with open(path, "r") as file:
            source = file.read()
            return source
    except Exception as e:
        print(e)

def create_shader(vertex, fragment):
    vertex_id = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_id, read_file(vertex))
    glCompileShader(vertex_id)

    frag_id = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(frag_id, read_file(fragment))
    glCompileShader(frag_id)

    program_id = glCreateProgram()
    glAttachShader(program_id, vertex_id)
    glAttachShader(program_id, frag_id)
    glLinkProgram(program_id)

    return program_id

def create_vao(vertices, faces):
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    flatern_array = []
    for vertice in vertices:
        flatern_array.extend(vertice)
    flatern_array = np.array(flatern_array, dtype=np.float32)
    # max_array = max(flatern_array)
    # min_array = min(flatern_array)
    # flatern_array = flatern_array * (1/(max_array-min_array))
    # print(flatern_array[:200]) 
    glBufferData(GL_ARRAY_BUFFER, flatern_array.nbytes, flatern_array, GL_STATIC_DRAW)

    ibo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
    flatern_array = []
    for face in faces:
        flatern_array.extend(face)
    flatern_array = np.array(flatern_array, dtype=np.uint32)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, flatern_array.nbytes, flatern_array, GL_STATIC_DRAW)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

    # glEnableVertexAttribArray(1)
    # glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))

    glBindVertexArray(0)

    return vao

def axis():
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    array = np.array([
        -1.5,  0,  0,
         1.5,  0,  0,
         0, -1.5,  0,
         0,  1.5,  0,
         0,  0, -1.5,
         0,  0,  1.5 
    ], dtype=np.float32)
    glBufferData(GL_ARRAY_BUFFER, array.nbytes, array, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindVertexArray(0)

    return vao

def create_texture(image_path, number):
    image = pygame.image.load(image_path)
    # image = self.load_image("./textures/wall.jpg")
    image = pygame.transform.flip(image, flip_x=False, flip_y=True)
    glActiveTexture(GL_TEXTURE0 + number)
    texture_id = glGenTextures(1)
    
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, 
        image.get_width(),
        image.get_height(),
        0, GL_RGB, GL_UNSIGNED_BYTE, pygame.image.tostring(image, "RGB"))

    return texture_id

def load_obj(filename):
    vertices, faces = [], []
    with open(filename, "r") as file:
        for line in file:
            if line.startswith("v"):
                #v 1 1 1\n
                vertices.append([float(i) for i in line.strip("\n").split()[1:]] + [1])
            elif line.startswith("f"):
                #f 1//1 1//1 1//1 1//1\n
                faces.append([int(i.split("/")[0]) - 1 for i in line.strip("\n").split()[1:]])
    return vertices, faces 

WIDTH, HEIGTH, FPS = 600, 600, 60
pygame.init()
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 6)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
win = pygame.display.set_mode((WIDTH, HEIGTH), flags=OPENGL | DOUBLEBUF)
print("Version:", glGetString(GL_VERSION))
print("Shader version:", glGetString(GL_SHADING_LANGUAGE_VERSION))
clock = pygame.time.Clock()
# vertices, faces = load_obj("../bugatti/bugatti.obj")
# vertices, faces = load_obj("../../learn_python/Software_3D_engine-main/resources/t_34_obj.obj")
# vertices, faces = load_obj("../MI28.obj")
vertices = [
    [0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, -0.5, -0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5]
]

faces = [
    [6, 4, 7],
    [6, 5, 4],
    [5, 0, 4],
    [5, 1, 0],
    [2, 7, 3],
    [2, 6, 7],
    [0, 2, 3],
    [0, 1, 2],
    [3, 7, 0],
    [7, 4, 0],
    [1, 5, 6],
    [1, 6, 2]
]

len_faces = len(faces)
print("vertices:", len(vertices))
print("faces:", len(faces))
vao = create_vao(vertices, faces)
vao1 = axis()
program = create_shader("./shaders/triangle.vert", "./shaders/triangle.frag")
# glFrontFace(GL_CW)
glLineWidth(2)
glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
# glFrontFace(GL_CCW)
glEnable(GL_DEPTH_TEST)
# glEnable(GL_CULL_FACE)
# glCullFace(GL_BACK)
# glOrtho(-10, 10, -10, 10, -10, 10)
glUseProgram(program)
glBindVertexArray(vao)
uniform = glGetUniformLocation(program, "translate")
uniform1 = glGetUniformLocation(program, "scale")
uniform2 = glGetUniformLocation(program, "rotate")
x = y = z = 0
sx = sy = sz = 1
angle_x = angle_y = angle_z = 30
count = 0
while True:
    pygame.display.set_caption(f"FPS: {clock.get_fps()}")
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    key = pygame.key.get_pressed()
    if key[K_RIGHT]:
        x += 0.01
    if key[K_LEFT]:
        x -= 0.01
    if key[K_UP]:
        y += 0.01
    if key[K_DOWN]:
        y -= 0.01
    if key[K_q]:
        sx += 0.01
        sy += 0.01
        sz += 0.01
    if key[K_e]:
        sx -= 0.01
        sy -= 0.01
        sz -= 0.01
    if key[K_d]:
        angle_x -= 0.02
        angle_y -= 0.02
        angle_z -= 0.02
    if key[K_a]:
        angle_x += 0.02
        angle_y += 0.02
        angle_z += 0.02
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # glClear(GL_COLOR_BUFFER_BIT)

    glUniformMatrix4fv(uniform, 1, GL_TRUE, translate(0, 0, 0))
    glUniformMatrix4fv(uniform1, 1, GL_TRUE, scale(1, 1, 1))
    glUniformMatrix4fv(uniform2, 1, GL_TRUE, rotate_x(0) @ rotate_y(0) @ rotate_z(0))
    glBindVertexArray(vao1)
    glDrawArrays(GL_LINES, 0, 6)
    glUniformMatrix4fv(uniform, 1, GL_TRUE, translate(x, y, z))
    glUniformMatrix4fv(uniform1, 1, GL_TRUE, scale(sx, sy, sz))
    glUniformMatrix4fv(uniform2, 1, GL_TRUE, rotate_x(angle_x) @ rotate_y(angle_y) @ rotate_z(angle_z))
    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, 3 * len_faces, GL_UNSIGNED_INT, None)
    glBindVertexArray(vao1)
    glDrawArrays(GL_LINES, 0, 6)

    angle_x += 0.007
    angle_y += 0.007
    angle_z += 0.007
    
    pygame.display.flip()
    clock.tick(FPS)
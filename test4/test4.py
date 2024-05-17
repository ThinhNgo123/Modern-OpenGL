import pygame, sys, math
from pygame.locals import *
from OpenGL.GL import *
import numpy as np
import ctypes 
import glm
from OpenGL.GL.shaders import compileShader, compileProgram

def indentity():
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def translate(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
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
        [1,            0,           0, 0],
        [0,  math.cos(a), math.sin(a), 0],
        [0, -math.sin(a), math.cos(a), 0],
        [0,            0,           0, 1]
    ]).transpose()

def rotate_y(a):
    return np.array([
        [math.cos(a), 0, -math.sin(a), 0],
        [          0, 1,            0, 0],
        [math.sin(a), 0,  math.cos(a), 0],
        [          0, 0,            0, 1]
    ]).transpose()

def rotate_z(a):
    return np.array([
        [ math.cos(a), math.sin(a), 0, 0],
        [-math.sin(a), math.cos(a), 0, 0],
        [           0,           0, 1, 0],
        [           0,           0, 0, 1]
    ]).transpose()

def rotate(alpha, beta, gamma):
    return rotate_x(alpha) @ rotate_y(beta) @ rotate_z(gamma)

def magnitude(vec):
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)

def normalize(vec):
    mag = magnitude(vec)
    if mag == 0:
        raise Exception("Length equal zero")
    return (vec[0] / mag, vec[1] / mag, vec[2] / mag)

def cross(vec1, vec2):
    return (vec1[1] * vec2[2] - vec2[1] * vec1[2],
            vec1[2] * vec2[0] - vec1[0] * vec2[2],
            vec1[0] * vec2[1] - vec2[0] * vec1[1])

def look_at(eye, center, up):
    direction = normalize((eye[0] - center[0], eye[1] - center[1], eye[2] - center[2]))
    right = normalize(cross(up, direction))
    new_up = normalize(cross(direction, right))
    return np.array([
        [    right[0],     right[1],     right[2], 0],
        [   new_up[0],    new_up[1],    new_up[2], 0],
        [direction[0], direction[1], direction[2], 0],
        [           0,            0,            0, 1]
    ]) @ \
    np.array([
        [1, 0, 0, -center[0]],
        [0, 1, 0, -center[1]],
        [0, 0, 1, -center[2]],
        [0, 0, 0,          1]
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
    status = glGetShaderiv(vertex_id, GL_COMPILE_STATUS)
    if not status:
        print(glGetShaderInfoLog(vertex_id))

    frag_id = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(frag_id, read_file(fragment))
    glCompileShader(frag_id)
    status = glGetShaderiv(frag_id, GL_COMPILE_STATUS)
    if not status:
        print(glGetShaderInfoLog(frag_id))

    program_id = glCreateProgram()
    glAttachShader(program_id, vertex_id)
    glAttachShader(program_id, frag_id)
    glLinkProgram(program_id)
    status = glGetProgramiv(program_id, GL_LINK_STATUS)
    if not status:
        print(glGetProgramInfoLog(program_id))

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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

    glBindVertexArray(0)

    return vao

def axis():
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    array = np.array([
        -0.8,  0,  0,
         0.8,  0,  0,
         0, -0.8,  0,
         0,  0.8,  0,
         0,  0, -0.8,
         0,  0,  0.8 
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

def get_model_matrix(
        translate=indentity(), 
        rotate=indentity(), 
        scale=indentity()
    ):
    # print("translate")
    # print(translate)
    # print("rotate")
    # print(rotate)
    return translate @ rotate @ scale

WIDTH, HEIGTH, FPS = 600, 600, 60
pygame.init()
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 6)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
win = pygame.display.set_mode((WIDTH, HEIGTH), flags=OPENGL | DOUBLEBUF)
# pygame.event.set_grab(True)
# pygame.mouse.set_visible(False)
print("Version:", glGetString(GL_VERSION))
print("Shader version:", glGetString(GL_SHADING_LANGUAGE_VERSION))
clock = pygame.time.Clock()
# vertices, faces = load_obj("../bugatti/bugatti.obj")
# vertices, faces = load_obj("../../learn_python/Software_3D_engine-main/resources/t_34_obj.obj")
# vertices, faces = load_obj("../MI28.obj")
vertices = [
    # position      # color
    [0.5, 0.5, 0.5, 1, 0, 0],
    [0.5, -0.5, 0.5, 0, 1, 0],
    [-0.5, -0.5, 0.5, 0, 0, 1],
    [-0.5, 0.5, 0.5, 1, 1, 0],
    [0.5, 0.5, -0.5, 0, 1, 1],
    [0.5, -0.5, -0.5, 1, 0, 1],
    [-0.5, -0.5, -0.5, 1, 1, 1],
    [-0.5, 0.5, -0.5, 0.5, 0.5, 0]
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
# glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
# glFrontFace(GL_CCW)
glEnable(GL_DEPTH_TEST)
glEnable(GL_CULL_FACE)
glCullFace(GL_BACK)
# glOrtho(-10, 10, -10, 10, -10, 10)
glUseProgram(program)
glBindVertexArray(vao)
model_location = glGetUniformLocation(program, "u_model")
view_location = glGetUniformLocation(program, "u_view")
u_model = indentity()
eye = [0, 0, 3]
center = [0, 0, 0]
up = [0, 1, 0]
while True:
    delta_time = 1 / clock.tick(FPS)
    pygame.display.set_caption(f"FPS: {clock.get_fps()}")
    for event in pygame.event.get():
        if (event.type == QUIT) or (event.type == KEYDOWN and event.key == K_ESCAPE):
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION:
            print(pygame.mouse.get_rel())
    key = pygame.key.get_pressed()
    if key[K_RIGHT]:
        eye[0] += 0.01
        # center[0] += 0.01
    if key[K_LEFT]:
        eye[0] -= 0.01
        # center[0] -= 0.01
    if key[K_UP]:
        eye[1] += 0.01
        # center[1] += 0.01
    if key[K_DOWN]:
        eye[1] -= 0.01
        # center[1] -= 0.01
    if key[K_e]:
        eye[2] += 0.01
        # center[2] += 0.01
    if key[K_q]:
        eye[2] -= 0.01
        # center[2] -= 0.01
    if key[K_a]:
        up[0] += 0.01
    if key[K_d]:
        up[0] -= 0.01
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # glClear(GL_COLOR_BUFFER_BIT)

    u_model = get_model_matrix(translate(0, 0, 0), rotate(0, 0, 0), scale(1, 1, 1))
    # print(x, y, z)
    u_view = look_at(eye, center, up)
    # print("model", u_model)
    print("view", u_view)
    glUniformMatrix4fv(model_location, 1, GL_FALSE, u_model)
    glUniformMatrix4fv(view_location, 1, GL_FALSE, u_view)
    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, 3 * len_faces, GL_UNSIGNED_INT, None)
    # glBindVertexArray(vao1)
    # glDrawArrays(GL_LINES, 0, 6)

    # angle_x += 0.007
    # angle_y += 0.007
    # angle_z += 0.01
    
    pygame.display.flip()
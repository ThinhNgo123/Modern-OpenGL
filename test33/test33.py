import pygame, sys, math
from pygame.locals import *
from OpenGL.GL import *
from PIL import Image
import numpy as np
import ctypes 
from shader import Shader 
from model import Model

def rad(degree):
    return degree * math.pi / 180

def deg(radian):
    return radian * 180 / math.pi

def indentity():
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def translate(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def scale(x, y, z):
    return np.array([
        [x, 0, 0, 0],
        [0, y, 0, 0],
        [0, 0, z, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def rotate_x(a):
    return np.array([
        [1,           0,            0, 0],
        [0, math.cos(a), -math.sin(a), 0],
        [0, math.sin(a),  math.cos(a), 0],
        [0,           0,            0, 1]
    ], dtype=np.float32)

def rotate_y(a):
    return np.array([
        [ math.cos(a), 0, math.sin(a), 0],
        [           0, 1,           0, 0],
        [-math.sin(a), 0, math.cos(a), 0],
        [           0, 0,           0, 1]
    ], dtype=np.float32)

def rotate_z(a):
    return np.array([
        [math.cos(a), -math.sin(a), 0, 0],
        [math.sin(a),  math.cos(a), 0, 0],
        [          0,            0, 1, 0],
        [          0,            0, 0, 1]
    ], dtype=np.float32)

def rotate(alpha, beta, gamma):
    return rotate_x(alpha) @ rotate_y(beta) @ rotate_z(gamma)

def ortho(left, right, bottom, top, near, far):
    return np.array([
        [2 / (right - left),                  0,                0, - (right + left) / (right - left)],
        [                 0, 2 / (top - bottom),                0, - (top + bottom) / (top - bottom)],
        [                 0,                  0, 2 / (near - far),     - (far + near) / (far - near)],
        [                 0,                  0,                0,                                 1]
    ], dtype=np.float32)

def frustum(l, r, b, t, n, f):
    assert (0 < n < f), "Near, far < 0 or near > far"
    return np.array([
        [(2 * n) / (r - l),                 0,  (r + l) / (r - l),                      0],
        [                0, (2 * n) / (t - b),  (t + b) / (t - b),                      0],
        [                0,                 0, -(f + n) / (f - n), (-2 * f * n) / (f - n)],
        [                0,                 0,                 -1,                      0]
    ], dtype=np.float32)

def perspective(fov_radian, aspect_ratio, near, far):
    h = 2 * near * math.tan(fov_radian / 2)
    w = aspect_ratio * h
    return frustum(- w / 2, w / 2, - h / 2, h / 2, near, far)

def magnitude(vec):
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)

def normalize(vec):
    mag = magnitude(vec)
    if mag == 0:
        raise Exception("Length equal zero")
    return [vec[0] / mag, vec[1] / mag, vec[2] / mag]

def cross(vec1, vec2):
    return [vec1[1] * vec2[2] - vec2[1] * vec1[2],
            vec1[2] * vec2[0] - vec1[0] * vec2[2],
            vec1[0] * vec2[1] - vec2[0] * vec1[1]]

def look_at(eye, center, up):
    direction = normalize((eye[0] - center[0], eye[1] - center[1], eye[2] - center[2]))
    right = normalize(cross(up, direction))
    new_up = normalize(cross(direction, right))
    return np.array([
        [    right[0],     right[1],     right[2], 0],
        [   new_up[0],    new_up[1],    new_up[2], 0],
        [direction[0], direction[1], direction[2], 0],
        [           0,            0,            0, 1]
    ], dtype=np.float32) @ \
    np.array([
        [1, 0, 0, -eye[0]],
        [0, 1, 0, -eye[1]],
        [0, 0, 1, -eye[2]],
        [0, 0, 0,       1]
    ], dtype=np.float32)

def look_at_no_translate(eye, center, up):
    view_matrix = look_at(eye, center, up)
    # print(view_matrix[0][0])
    # view_matrix[3][0] = 0
    # view_matrix[3][1] = 0
    # view_matrix[3][2] = 0
    view_matrix[0][3] = 0
    view_matrix[1][3] = 0
    view_matrix[2][3] = 0
    # print(view_matrix)
    return view_matrix

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

def create_vao(vertices):
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    flatern_array = np.array(vertices, dtype=np.float32)
    glBufferData(GL_ARRAY_BUFFER, flatern_array.nbytes, flatern_array, GL_STATIC_DRAW)

    # ebo = glGenBuffers(1)
    # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    # flatern_array = np.array(faces, dtype=np.uint32)
    # glBufferData(GL_ELEMENT_ARRAY_BUFFER, flatern_array.nbytes, flatern_array, GL_STATIC_DRAW)

    size = np.float32().itemsize

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * size, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * size, ctypes.c_void_p(3 * size))

    # glEnableVertexAttribArray(2)
    # glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * size, ctypes.c_void_p(6 * size))

    glBindVertexArray(0)

    return vao

def create_skybox_vao(vertices):
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    flatern_array = np.array(vertices, dtype=np.float32)
    glBufferData(GL_ARRAY_BUFFER, flatern_array.nbytes, flatern_array, GL_STATIC_DRAW)

    # ebo = glGenBuffers(1)
    # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    # flatern_array = np.array(faces, dtype=np.uint32)
    # glBufferData(GL_ELEMENT_ARRAY_BUFFER, flatern_array.nbytes, flatern_array, GL_STATIC_DRAW)

    size = np.float32().itemsize

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * size, ctypes.c_void_p(0))

    # glEnableVertexAttribArray(1)
    # glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * size, ctypes.c_void_p(3 * size))

    # glEnableVertexAttribArray(2)
    # glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * size, ctypes.c_void_p(6 * size))

    glBindVertexArray(0)

    return vao

def axis(a=20):

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    array = np.array([
        -a, 0, 0,
        a, 0, 0,
        0, -a, 0,
        0, a, 0,
        0, 0, -a,
        0, 0, a 
    ], dtype=np.float32)
    glBufferData(GL_ARRAY_BUFFER, array.nbytes, array, GL_STATIC_DRAW)

    # ebo = glGenBuffers(1)
    # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    # array = np.array([
    #     0, 1, 3,
    #     1, 2, 3
    # ], dtype=np.float32)
    # glBufferData(GL_ELEMENT_ARRAY_BUFFER, array.nbytes, array, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindVertexArray(0)

    return vao

def create_light_vao():
    vao = glGenVertexArrays(1)
    vbo, ebo = glGenBuffers(2)
    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    vertices = np.array(
        [
             1, -1,  1,
             1,  1,  1,
            -1,  1,  1,
            -1, -1,  1,
             1, -1, -1,
             1,  1, -1,
            -1,  1, -1,
            -1, -1, -1,

        ],
    dtype=np.float32)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    indices = np.array([
        0, 1, 2,
        3, 0, 2,
        7, 5, 4,
        7, 6, 5,
        1, 6, 2, 
        1, 5, 6,
        0, 3, 7,
        0, 7, 4,
        0, 5, 1,
        0, 4, 5,
        7, 2, 6,
        7, 3, 2
    ], dtype=np.uint32)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * np.float32().itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    return vao

def create_texture(image_path: str, number):
    # image = pygame.image.load(image_path)
    ext_file = image_path[image_path.rfind(".")+1:]
    # print(ext_file)
    image = Image.open(image_path)
    # print(image.format)
    # if ext_file == "png":
    #     for i in range(image.width):
    #         for j in range(image.height):
    #             print(image.getpixel((i, j)), end=" ")
    #         print()
    #     print()

    # image = self.load_image("./textures/wall.jpg")
    # image = pygame.transform.flip(image, flip_x=False, flip_y=True)
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    glActiveTexture(GL_TEXTURE0 + number)
    texture_id = glGenTextures(1)
    
    glBindTexture(GL_TEXTURE_2D, texture_id)
    # glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT)
    # glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [1, 1, 0])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    
    if ext_file == "jpg":
        format = GL_RGB
    elif ext_file == "png":
        format = GL_RGBA
    else:
        format = GL_RGB
    # print(format)
    glTexImage2D(
        GL_TEXTURE_2D, 0, format, 
        # image.get_width(),
        image.width,
        # image.get_height(),
        image.height,
        # 0, GL_RGB, GL_UNSIGNED_BYTE, pygame.image.tostring(image, "RGB"))
        0, format, GL_UNSIGNED_BYTE, image.tobytes())
    glGenerateMipmap(GL_TEXTURE_2D)

    return number

def create_texture_wrap_clamp_edge(image_path: str, number):
    # image = pygame.image.load(image_path)
    ext_file = image_path[image_path.rfind(".")+1:]
    image = Image.open(image_path)
    # print(image.getpixel((0, 0)))

    # image = self.load_image("./textures/wall.jpg")
    # image = pygame.transform.flip(image, flip_x=False, flip_y=True)
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    glActiveTexture(GL_TEXTURE0 + number)
    texture_id = glGenTextures(1)
    
    glBindTexture(GL_TEXTURE_2D, texture_id)
    # glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    # glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [1, 1, 0])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    
    if ext_file == "png":
        format = GL_RGBA
    else:
        format = GL_RGB
    
    glTexImage2D(
        GL_TEXTURE_2D, 0, format, 
        # image.get_width(),
        image.width,
        # image.get_height(),
        image.height,
        # 0, GL_RGB, GL_UNSIGNED_BYTE, pygame.image.tostring(image, "RGB"))
        0, format, GL_UNSIGNED_BYTE, image.tobytes())
    glGenerateMipmap(GL_TEXTURE_2D)

    return number

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

def create_fbo():
    return glGenFramebuffers(1)

def frame_buffer_to_texture(number: int):
    fbo = create_fbo()
    texture = glGenTextures(1)
    rbo = glGenRenderbuffers(1)

    glActiveTexture(GL_TEXTURE0 + number)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGTH, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    # glGenerateMipmap(GL_TEXTURE_2D)

    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, WIDTH, HEIGTH)
    # glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_STENCIL, WIDTH, HEIGTH)

    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)
    # glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo)

    if not (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE):
        raise Exception("Create frame buffer failed")
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    return fbo, number

def create_cube_map_texture(path, number, format="jpg"):
    image_faces = [
        "right",
        "left",
        "top",
        "bottom",
        "front",
        "back"
    ]

    cube_texture = glGenTextures(1)
    glActiveTexture(GL_TEXTURE0 + number)
    glBindTexture(GL_TEXTURE_CUBE_MAP, cube_texture)

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

    for i in range(6):
        image = Image.open(path + "/" + image_faces[i] + "." + format)

        glTexImage2D(
            #target,level,internalformat,width,height,border,format,type,pixels
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
            0,
            GL_RGB,
            image.width,
            image.height,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            image.tobytes()
        )
    return number

WIDTH, HEIGTH, FPS = 1200, 800, 200
pygame.init()
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 6)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
# pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
# pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
# pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 1)
win = pygame.display.set_mode((WIDTH, HEIGTH), flags=OPENGL | DOUBLEBUF)

# glEnable(GL_MULTISAMPLE)
glEnable(GL_DEPTH_TEST)
# glEnable(GL_STENCIL_TEST)
# glEnable(GL_BLEND)

# glFrontFace(GL_CCW)
# glFrontFace(GL_CW)
# glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
# glEnable(GL_CULL_FACE)
# glCullFace(GL_FRONT)
# glCullFace(GL_BACK)
glLineWidth(2)
# pygame.event.set_grab(True)
# pygame.mouse.set_visible(False)
pygame.mouse.set_pos(WIDTH / 2, HEIGTH / 2)
print("Version:", glGetString(GL_VERSION))
print("Shader version:", glGetString(GL_SHADING_LANGUAGE_VERSION))
clock = pygame.time.Clock()
current_frame = last_frame = 0
# vertices, faces = load_obj("../bugatti/bugatti.obj")
# vertices, faces = load_obj("../../learn_python/Software_3D_engine-main/resources/t_34_obj.obj")
# vertices, faces = load_obj("../MI28.obj")

cube_vertices = [
    # positions        normal
    -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
     0.5, -0.5, -0.5,  0.0,  0.0, -1.0, 
     0.5,  0.5, -0.5,  0.0,  0.0, -1.0, 
     0.5,  0.5, -0.5,  0.0,  0.0, -1.0, 
    -0.5,  0.5, -0.5,  0.0,  0.0, -1.0, 
    -0.5, -0.5, -0.5,  0.0,  0.0, -1.0, 

    -0.5, -0.5,  0.5,  0.0,  0.0, 1.0,
     0.5, -0.5,  0.5,  0.0,  0.0, 1.0,
     0.5,  0.5,  0.5,  0.0,  0.0, 1.0,
     0.5,  0.5,  0.5,  0.0,  0.0, 1.0,
    -0.5,  0.5,  0.5,  0.0,  0.0, 1.0,
    -0.5, -0.5,  0.5,  0.0,  0.0, 1.0,

    -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,
    -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,
    -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,
    -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,
    -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,
    -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,

     0.5,  0.5,  0.5,  1.0,  0.0,  0.0,
     0.5,  0.5, -0.5,  1.0,  0.0,  0.0,
     0.5, -0.5, -0.5,  1.0,  0.0,  0.0,
     0.5, -0.5, -0.5,  1.0,  0.0,  0.0,
     0.5, -0.5,  0.5,  1.0,  0.0,  0.0,
     0.5,  0.5,  0.5,  1.0,  0.0,  0.0,

    -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
     0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
     0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
     0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
    -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
    -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,

    -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
     0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
     0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
     0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
    -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
    -0.5,  0.5, -0.5,  0.0,  1.0,  0.0
]

plane_vertices = [
    # positions          // texture Coords (note we set these higher than 1 (together with GL_REPEAT as texture wrapping mode). this will cause the loor texture to repeat)
    5.0, -0.5,  5.0,  2.0, 0.0,
    -5.0, -0.5, -5.0,  0.0, 2.0,
    -5.0, -0.5,  5.0,  0.0, 0.0,

    5.0, -0.5,  5.0,  2.0, 0.0,
    5.0, -0.5, -5.0,  2.0, 2.0,								
    -5.0, -0.5, -5.0,  0.0, 2.0
]

screen_vertices = [
    -1, -1, 0, 0, 0, 
     1, -1, 0, 1, 0,
    -1,  1, 0, 0, 1,
     1, -1, 0, 1, 0,
     1,  1, 0, 1, 1,
    -1,  1, 0, 0, 1
]

skybox_vertices = [
    # positions          
    -1.0,  1.0, -1.0,
    -1.0, -1.0, -1.0,
     1.0, -1.0, -1.0,
     1.0, -1.0, -1.0,
     1.0,  1.0, -1.0,
    -1.0,  1.0, -1.0,

    -1.0, -1.0,  1.0,
    -1.0, -1.0, -1.0,
    -1.0,  1.0, -1.0,
    -1.0,  1.0, -1.0,
    -1.0,  1.0,  1.0,
    -1.0, -1.0,  1.0,

     1.0, -1.0, -1.0,
     1.0, -1.0,  1.0,
     1.0,  1.0,  1.0,
     1.0,  1.0,  1.0,
     1.0,  1.0, -1.0,
     1.0, -1.0, -1.0,

    -1.0, -1.0,  1.0,
    -1.0,  1.0,  1.0,
     1.0,  1.0,  1.0,
     1.0,  1.0,  1.0,
     1.0, -1.0,  1.0,
    -1.0, -1.0,  1.0,

    -1.0,  1.0, -1.0,
     1.0,  1.0, -1.0,
     1.0,  1.0,  1.0,
     1.0,  1.0,  1.0,
    -1.0,  1.0,  1.0,
    -1.0,  1.0, -1.0,

    -1.0, -1.0, -1.0,
    -1.0, -1.0,  1.0,
     1.0, -1.0, -1.0,
     1.0, -1.0, -1.0,
    -1.0, -1.0,  1.0,
     1.0, -1.0,  1.0
]

cube_vao = create_vao(cube_vertices)
# plane_vao = create_vao(plane_vertices)
skybox_vao = create_skybox_vao(skybox_vertices)

uniform_matrix_block = glGenBuffers(1)
glBindBuffer(GL_UNIFORM_BUFFER, uniform_matrix_block)
glBufferData(GL_UNIFORM_BUFFER, 4 * 4 * 4 * 2 + 4 * 3, None, GL_STATIC_DRAW)
glBindBuffer(GL_UNIFORM_BUFFER, 0)
glBindBufferBase(GL_UNIFORM_BUFFER, 0, uniform_matrix_block)

model = Model("../backpack/backpack.obj")

reflection_shader = Shader(
    "./shaders/cubemap_reflect.vert",
    "./shaders/cubemap_reflect.frag"
)

refraction_shader = Shader(
    "./shaders/cubemap_refraction.vert",
    "./shaders/cubemap_refraction.frag"
)

skybox_shader = Shader(
    "./shaders/skybox.vert",
    "./shaders/skybox.frag"
)

# reflection_matrix_location = glGetUniformBlockIndex(reflection_shader.program_id, "Matrices")
# refraction_matrix_location = glGetUniformBlockIndex(refraction_shader.program_id, "Matrices")

# glUniformBlockBinding(reflection_shader.program_id, reflection_matrix_location, 0)
# glUniformBlockBinding(refraction_shader.program_id, refraction_matrix_location, 0)


# texture1 = create_texture("./textures/Tileable marble floor tile texture (6).jpg", 0)
# texture2 = create_texture("./textures/metal-texture-25.jpg", 1)
texture3 = create_cube_map_texture("../skybox", 2)

# screen_fbo, texture3 = frame_buffer_to_texture(2)

eye = [0, 0, 10]
center = [0, 0, 0]
up = [0, 1, 0]
forward = normalize([center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]])
right = normalize(cross(forward, up))
up = normalize(cross(right, forward))
# sx = sy = sz = 1
# print(forward, right, up)
angle_x = angle_y = angle_z = 0.1
FOV = 45
aspect_ratio = WIDTH / HEIGTH
camera_speed = 2.5
yaw = pitch = 0
sensitivity = 0.1
mode_draw = False
cutoff = 12.5
outer_cutoff = 13.5
offset = outer_cutoff - cutoff
shininess = 10
# glEnable(GL_SCISSOR_TEST)
count = 1
mouse_motion_allow = False
offset_inverse_sampling = 300
screen_small_pos = [0, 0]

while True:
    clock.tick(FPS)
    # clock.tick_busy_loop(FPS)
    current_frame = pygame.time.get_ticks()
    delta_time = (current_frame - last_frame) * 0.001
    last_frame = current_frame
    # delta_time = 1 / clock.tick(FPS)
    # print(delta_time)
    pygame.display.set_caption(f"FPS: {clock.get_fps()}")
    for event in pygame.event.get():
        if (event.type == QUIT) or (event.type == KEYDOWN and event.key == K_ESCAPE):
            # gui.quit()
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION:
            rel_x, rel_y = pygame.mouse.get_rel()
            # print(rel_x, rel_y)
            if mouse_motion_allow:
                yaw += rel_x * sensitivity
                pitch -= rel_y * sensitivity
                forward = normalize([
                    math.sin(rad(yaw)) * math.cos(rad(pitch)),
                    math.sin(rad(pitch)),
                    -math.cos(rad(yaw)) * math.cos(rad(pitch))
                ])
                right = normalize(cross(forward, [0, 1, 0]))
                up = normalize(cross(right, forward))
            # print(center)
        if event.type == MOUSEWHEEL:
            # FOV -= event.precise_y * 2
            # if FOV <= 1:
            #     FOV = 1
            # elif FOV >= 45:
            #     FOV = 45
            shininess += event.precise_y * 2
            if shininess <= 0:
                shininess = 0.01
            print("shininess:", shininess)
        if event.type == KEYDOWN:
            if event.key == K_p:
                mode_draw = not mode_draw
            if event.key == K_KP_1:
                count += 1
                print(count)
            if event.key == K_KP_2:
                count -= 1
                print(count)
            if event.key == K_SPACE:
                mouse_motion_allow = not mouse_motion_allow

    key = pygame.key.get_pressed()
    if key[K_d]:
        eye = [
            eye[0] + right[0] * camera_speed * delta_time,
            eye[1] + right[1] * camera_speed * delta_time,
            eye[2] + right[2] * camera_speed * delta_time,
        ]
    if key[K_a]:
        eye = [
            eye[0] - right[0] * camera_speed * delta_time,
            eye[1] - right[1] * camera_speed * delta_time,
            eye[2] - right[2] * camera_speed * delta_time,
        ]
    if key[K_w]:
        eye = [
            eye[0] + up[0] * camera_speed * delta_time,
            eye[1] + up[1] * camera_speed * delta_time,
            eye[2] + up[2] * camera_speed * delta_time,
        ]
    if key[K_s]:
        eye = [
            eye[0] - up[0] * camera_speed * delta_time,
            eye[1] - up[1] * camera_speed * delta_time,
            eye[2] - up[2] * camera_speed * delta_time,
        ]
    if key[K_e]:
        eye = [
            eye[0] + forward[0] * camera_speed * delta_time * 2,
            eye[1] + forward[1] * camera_speed * delta_time * 2,
            eye[2] + forward[2] * camera_speed * delta_time * 2,
        ]
        # eye[1] = 0
    if key[K_q]:
        eye = [
            eye[0] - forward[0] * camera_speed * delta_time * 2,
            eye[1] - forward[1] * camera_speed * delta_time * 2,
            eye[2] - forward[2] * camera_speed * delta_time * 2,
        ]
        # eye[1] = 0
    if key[K_RIGHT]:
        screen_small_pos[0] += camera_speed * delta_time
        print(screen_small_pos)
    if key[K_LEFT]:
        screen_small_pos[0] -= camera_speed * delta_time
        print(screen_small_pos)
    if key[K_UP]:
        screen_small_pos[1] += camera_speed * delta_time
        print(screen_small_pos)
    if key[K_DOWN]:
        screen_small_pos[1] -= camera_speed * delta_time
        print(screen_small_pos)

    # u_model = get_model_matrix(rotate=rotate(rad(-55), 0, 0))
    # 1
    # glViewport(0, HEIGTH // 2, WIDTH // 2, HEIGTH // 2)
    # glScissor(0, HEIGTH // 2, WIDTH // 2, HEIGTH // 2)
    # glClearColor(0.1, 0.1, 0.1, 1)
    # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
    if mode_draw:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    else:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    # glDepthMask(GL_TRUE)
    glDepthFunc(GL_LEQUAL)

    # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    

    u_view = look_at(eye, [eye[0] + forward[0], eye[1] + forward[1], eye[2] + forward[2]], up)
    u_view_skybox = look_at_no_translate(eye, [eye[0] + forward[0], eye[1] + forward[1], eye[2] + forward[2]], up)
    u_proj = perspective(rad(FOV), aspect_ratio, 0.1, 100)
    # reflection_shader.use()
    # reflection_shader.setMatrix4("u_view", u_view)
    # reflection_shader.setMatrix4("u_proj1", u_proj)
    # reflection_shader.setFloat3("eye", *eye)

    # refraction_shader.use()
    # refraction_shader.setMatrix4("u_view", u_view)
    # shader.setMatrix4("u_proj", u_proj)
    # refraction_shader.setFloat3("eye", *eye)

    eye_array = np.array(eye, dtype=np.float32)
    glBindBuffer(GL_UNIFORM_BUFFER, uniform_matrix_block)
    glBufferSubData(GL_UNIFORM_BUFFER, 0, u_view.nbytes, u_view.transpose())
    glBufferSubData(GL_UNIFORM_BUFFER, u_view.nbytes, u_proj.nbytes, u_proj.transpose())
    # glBufferSubData(GL_UNIFORM_BUFFER, 64, u_proj.nbytes, u_proj.transpose())
    glBufferSubData(GL_UNIFORM_BUFFER, u_view.nbytes + u_proj.nbytes, eye_array.nbytes, eye_array)

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.1, 0.1, 0.1, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # glDepthMask(False)
    skybox_shader.use()
    skybox_shader.setMatrix4("u_view", u_view_skybox)
    skybox_shader.setMatrix4("u_proj", u_proj)
    skybox_shader.setInt("skyboxTexture", texture3)
    glBindVertexArray(skybox_vao)
    glDrawArrays(GL_TRIANGLES, 0, 36)
    # glDepthMask(True)

    # shader.use()
    # u_model = get_model_matrix(translate=translate(0, 0, 0))
    # shader.setMatrix4("u_model", u_model)
    # shader.setInt("texture1", texture2)
    # glBindVertexArray(plane_vao)
    # glDrawArrays(GL_TRIANGLES, 0, 6)
    
    reflection_shader.use()
    u_model = get_model_matrix(translate=translate(-5, 0.001, -5), scale=scale(2, 2, 2))
    reflection_shader.setMatrix4("u_model", u_model)
    reflection_shader.setInt("texture1", texture3)
    # glBindVertexArray(cube_vao)
    # glDrawArrays(GL_TRIANGLES, 0, 36)
    model.draw(reflection_shader)

    refraction_shader.use()
    u_model = get_model_matrix(translate=translate(5, 0.001, -5), scale=scale(2, 2, 2))
    refraction_shader.setMatrix4("u_model", u_model)
    refraction_shader.setInt("texture1", texture3)
    # # glBindVertexArray(cube_vao)
    # # glDrawArrays(GL_TRIANGLES, 0, 36)
    model.draw(refraction_shader)

    # shader.use()
    # u_model = get_model_matrix(translate=translate(2, 0.001, 0))
    # shader.setMatrix4("u_model", u_model)
    # shader.setInt("texture1", texture1)
    # glBindVertexArray(cube_vao)
    # glDrawArrays(GL_TRIANGLES, 0, 36)
        
    # angle_x += 0.007
    # angle_y += 0.007
    # angle_z += 0.01

    pygame.display.flip()
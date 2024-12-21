import random
import pygame, sys, math
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.extensions import hasExtension
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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * size, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * size, ctypes.c_void_p(3 * size))

    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * size, ctypes.c_void_p(6 * size))

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

def create_texture(image_path: str):
    # image = pygame.image.load(image_path)
    ext_file = image_path[image_path.rfind(".")+1:]
    # print(ext_file)
    image = Image.open(image_path)
    print(image.format)
    # if ext_file == "png":
    #     for i in range(image.width):
    #         for j in range(image.height):
    #             print(image.getpixel((i, j)), end=" ")
    #         print()
    #     print()

    # image = self.load_image("./textures/wall.jpg")
    # image = pygame.transform.flip(image, flip_x=False, flip_y=True)
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    # glActiveTexture(GL_TEXTURE0 + number)
    texture_id = glGenTextures(1)
    
    glBindTexture(GL_TEXTURE_2D, texture_id)
    # glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    # glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [1, 1, 0])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_REPEAT)
    
    if ext_file == "jpg":
        format = GL_RGB
    elif ext_file == "png":
        format = GL_RGBA
    else:
        format = GL_RGB
    # print(format)

    # format = GL_RGB
    glTexImage2D(
        GL_TEXTURE_2D, 0, format, 
        # image.get_width(),
        image.width,
        # image.get_height(),
        image.height,
        # 0, GL_RGB, GL_UNSIGNED_BYTE, pygame.image.tostring(image, "RGB"))
        0, format, GL_UNSIGNED_BYTE, image.tobytes())
    glGenerateMipmap(GL_TEXTURE_2D)

    return texture_id

def create_texture_wrap_clamp_edge(image_path: str):
    # image = pygame.image.load(image_path)
    ext_file = image_path[image_path.rfind(".")+1:]
    image = Image.open(image_path)
    # print(image.getpixel((0, 0)))

    # image = self.load_image("./textures/wall.jpg")
    # image = pygame.transform.flip(image, flip_x=False, flip_y=True)
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    # glActiveTexture(GL_TEXTURE0 + number)
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

def create_fbo():
    return glGenFramebuffers(1)

def frame_buffer_to_texture():
    fbo = create_fbo()
    texture = glGenTextures(1)
    rbo = glGenRenderbuffers(1)

    # glActiveTexture(GL_TEXTURE0 + number)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    # glGenerateMipmap(GL_TEXTURE_2D)

    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, WIDTH, HEIGHT)
    # glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_STENCIL, WIDTH, HEIGTH)

    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)
    # glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo)

    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise Exception("Create frame buffer failed")
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    return fbo, texture

def frame_buffer_to_texture_multisample():
    fbo = create_fbo()
    texture = glGenTextures(1)
    rbo = glGenRenderbuffers(1)
    
    # glActiveTexture(GL_TEXTURE0 + number)
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, texture)
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGB, WIDTH, HEIGHT, GL_TRUE)
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0)

    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH24_STENCIL8, WIDTH, HEIGHT)
    # glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH_COMPONENT24, WIDTH, HEIGHT)
    # glBindRenderbuffer(GL_RENDERBUFFER, 0)

    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, texture, 0)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo)
    # glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)

    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise Exception("Create frame buffer multisample failed")
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    return fbo, texture

def frame_buffer_only_depth(width: int, height: int):
    fbo = create_fbo()
    tex = glGenTextures(1)

    # glActiveTexture(GL_TEXTURE0 + number)
    glBindTexture(GL_TEXTURE_2D, tex)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, np.array([1, 1, 1, 1], dtype=np.float32))
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # glGenerateMipmap(GL_TEXTURE_2D)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)

    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex, 0)
    glReadBuffer(GL_NONE)
    glDrawBuffer(GL_NONE)

    if (not glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE):
        raise Exception("Create frame buffer failed")
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    return fbo, tex

def frame_buffer_only_cube_depth(width: int, height: int):
    fbo = create_fbo()
    tex = glGenTextures(1)

    texture_faces = [
        "right",
        "left",
        "top",
        "bottom",
        "back",
        "front"
    ]

    glBindTexture(GL_TEXTURE_CUBE_MAP, tex)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
   
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # glGenerateMipmap(GL_TEXTURE_2D)
    for i in range(len(texture_faces)):
        glTexImage2D(
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 
            0, 
            GL_DEPTH_COMPONENT, 
            width, 
            height, 
            0, 
            GL_DEPTH_COMPONENT, 
            GL_FLOAT, 
            None
        )

    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, tex, 0)
    glReadBuffer(GL_NONE)
    glDrawBuffer(GL_NONE)
    
    if (not glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE):
        raise Exception("Create frame buffer failed")
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    return fbo, tex

def frame_buffer_hdr(width: int, height: int):
    fbo = create_fbo()
    tex = glGenTextures(1)
    rbo = glGenRenderbuffers(1)

    # glActiveTexture(GL_TEXTURE0 + number)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, None)

    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)

    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)

    if (not glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE):
        raise Exception("Create frame buffer failed")
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glBindRenderbuffer(GL_RENDERBUFFER, 0)
    glBindTexture(GL_TEXTURE_2D, 0)

    return fbo, tex

def create_cube_map_texture(path, format="jpg"):
    image_faces = [
        "right",
        "left",
        "top",
        "bottom",
        "back",
        "front"
    ]

    cube_texture = glGenTextures(1)
    # glActiveTexture(GL_TEXTURE0 + number)
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
    return cube_texture

def debug_output(source, type, id, severity, length, message, user_param):
    print("Error!")

WIDTH, HEIGHT, FPS = 1200, 800, 120
pygame.init()
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 6)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
# pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FLAGS, pygame.GL_CONTEXT_DEBUG_FLAG)
# pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
# pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 4)
# pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 1)
win = pygame.display.set_mode((WIDTH, HEIGHT), flags=OPENGL | DOUBLEBUF)

# context_flags_success = glGetIntegerv(GL_CONTEXT_FLAGS)
# if (context_flags_success & GL_CONTEXT_FLAG_DEBUG_BIT):
#     print("Debug context created!")
#     glEnable(GL_DEBUG_OUTPUT)
#     glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS)
#     glDebugMessageCallback(GLDEBUGPROC(debug_output), None)
    # glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, None, GL_TRUE)


# glEnable(GL_MULTISAMPLE)
glEnable(GL_DEPTH_TEST)
# glEnable(GL_STENCIL_TEST)
# glEnable(GL_BLEND)

# glFrontFace(GL_CCW)
# glFrontFace(GL_CW)
# glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
glEnable(GL_CULL_FACE)
# glCullFace(GL_FRONT)
glCullFace(GL_BACK)
# glEnable(GL_SCISSOR_TEST)
glLineWidth(1)
# pygame.event.set_grab(True)
# pygame.mouse.set_visible(False)
pygame.mouse.set_pos(WIDTH / 2, HEIGHT / 2)
print("Version:", glGetString(GL_VERSION))
print("Shader version:", glGetString(GL_SHADING_LANGUAGE_VERSION))
print("Indirect supported:", hasExtension('GL_ARB_multi_draw_indirect'))
# print(wrapper)
# if hasattr(platform, 'GL'):
#     print("PyOpenGL_accelerate is active.")
# else:
#     print("PyOpenGL_accelerate is not active.")

clock = pygame.time.Clock()
current_frame = last_frame = 0
# vertices, faces = load_obj("../bugatti/bugatti.obj")
# vertices, faces = load_obj("../../learn_python/Software_3D_engine-main/resources/t_34_obj.obj")
# vertices, faces = load_obj("../MI28.obj")

quad_vertices = [
    # positions      colors
    -0.05,  0.05,  1.0, 0.0, 0.0,
     0.05, -0.05,  0.0, 1.0, 0.0,
    -0.05, -0.05,  0.0, 0.0, 1.0,

    -0.05,  0.05,  1.0, 0.0, 0.0,
     0.05, -0.05,  0.0, 1.0, 0.0,   
     0.05,  0.05,  0.0, 1.0, 1.0		    		
]

cube_vertices = [
    -1.0, -1.0, -1.0,  0.0,  0.0, -1.0, 0.0, 0.0, # bottom-let
     1.0,  1.0, -1.0,  0.0,  0.0, -1.0, 1.0, 1.0, # top-right
     1.0, -1.0, -1.0,  0.0,  0.0, -1.0, 1.0, 0.0, # bottom-right         
     1.0,  1.0, -1.0,  0.0,  0.0, -1.0, 1.0, 1.0, # top-right
    -1.0, -1.0, -1.0,  0.0,  0.0, -1.0, 0.0, 0.0, # bottom-let
    -1.0,  1.0, -1.0,  0.0,  0.0, -1.0, 0.0, 1.0, # top-let
    # ront ace
    -1.0, -1.0,  1.0,  0.0,  0.0,  1.0, 0.0, 0.0, # bottom-let
     1.0, -1.0,  1.0,  0.0,  0.0,  1.0, 1.0, 0.0, # bottom-right
     1.0,  1.0,  1.0,  0.0,  0.0,  1.0, 1.0, 1.0, # top-right
     1.0,  1.0,  1.0,  0.0,  0.0,  1.0, 1.0, 1.0, # top-right
    -1.0,  1.0,  1.0,  0.0,  0.0,  1.0, 0.0, 1.0, # top-let
    -1.0, -1.0,  1.0,  0.0,  0.0,  1.0, 0.0, 0.0, # bottom-let
    # let ace
    -1.0,  1.0,  1.0, -1.0,  0.0,  0.0, 1.0, 0.0, # top-right
    -1.0,  1.0, -1.0, -1.0,  0.0,  0.0, 1.0, 1.0, # top-let
    -1.0, -1.0, -1.0, -1.0,  0.0,  0.0, 0.0, 1.0, # bottom-let
    -1.0, -1.0, -1.0, -1.0,  0.0,  0.0, 0.0, 1.0, # bottom-let
    -1.0, -1.0,  1.0, -1.0,  0.0,  0.0, 0.0, 0.0, # bottom-right
    -1.0,  1.0,  1.0, -1.0,  0.0,  0.0, 1.0, 0.0, # top-right
    # right ace
     1.0,  1.0,  1.0,  1.0,  0.0,  0.0, 1.0, 0.0, # top-let
     1.0, -1.0, -1.0,  1.0,  0.0,  0.0, 0.0, 1.0, # bottom-right
     1.0,  1.0, -1.0,  1.0,  0.0,  0.0, 1.0, 1.0, # top-right         
     1.0, -1.0, -1.0,  1.0,  0.0,  0.0, 0.0, 1.0, # bottom-right
     1.0,  1.0,  1.0,  1.0,  0.0,  0.0, 1.0, 0.0, # top-let
     1.0, -1.0,  1.0,  1.0,  0.0,  0.0, 0.0, 0.0, # bottom-let     
    # bottom ace
    -1.0, -1.0, -1.0,  0.0, -1.0,  0.0, 0.0, 1.0, # top-right
     1.0, -1.0, -1.0,  0.0, -1.0,  0.0, 1.0, 1.0, # top-let
     1.0, -1.0,  1.0,  0.0, -1.0,  0.0, 1.0, 0.0, # bottom-let
     1.0, -1.0,  1.0,  0.0, -1.0,  0.0, 1.0, 0.0, # bottom-let
    -1.0, -1.0,  1.0,  0.0, -1.0,  0.0, 0.0, 0.0, # bottom-right
    -1.0, -1.0, -1.0,  0.0, -1.0,  0.0, 0.0, 1.0, # top-right
    # top ace
    -1.0,  1.0, -1.0,  0.0,  1.0,  0.0, 0.0, 1.0, # top-let
     1.0,  1.0,  1.0,  0.0,  1.0,  0.0, 1.0, 0.0, # bottom-right
     1.0,  1.0, -1.0,  0.0,  1.0,  0.0, 1.0, 1.0, # top-right     
     1.0,  1.0,  1.0,  0.0,  1.0,  0.0, 1.0, 0.0, # bottom-right
    -1.0,  1.0, -1.0,  0.0,  1.0,  0.0, 0.0, 1.0, # top-let
    -1.0,  1.0,  1.0,  0.0,  1.0,  0.0, 0.0, 0.0  # bottom-left 
]

plane_vertices = [
    # positions         #normals #texcoords
     25.0, -0.5,  25.0, 0, 1, 0, 25.0,  0.0,
    -25.0, -0.5,  25.0, 0, 1, 0,  0.0,  0.0,
    -25.0, -0.5, -25.0, 0, 1, 0,  0.0, 25.0,

     25.0, -0.5,  25.0, 0, 1, 0, 25.0,  0.0,
    -25.0, -0.5, -25.0, 0, 1, 0,  0.0, 25.0,
     25.0, -0.5, -25.0, 0, 1, 0, 25.0, 25.0
]

screen_vertices = [
    -1, -1, 0, 0, 0, 1, 0, 0, 
     1, -1, 0, 0, 0, 1, 1, 0,
    -1,  1, 0, 0, 0, 1, 0, 1,
     1, -1, 0, 0, 0, 1, 1, 0,
     1,  1, 0, 0, 0, 1, 1, 1,
    -1,  1, 0, 0, 0, 1, 0, 1
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

points = [
    -0.5,  0.5, 1.0, 0.0, 0.0, 
     0.5,  0.5, 0.0, 1.0, 0.0, 
     0.5, -0.5, 0.0, 0.0, 1.0,
    -0.5, -0.5, 1.0, 1.0, 0.0
]

plane_vao = create_vao(plane_vertices)
light_vao = create_light_vao()
cube_vao = create_vao(cube_vertices)
screen_vao = create_vao(screen_vertices)

# DEPTH_WIDTH = 1024
# DEPTH_HEIGHT = 1024
# depth_fbo, depth_map = frame_buffer_only_cube_depth(DEPTH_WIDTH, DEPTH_HEIGHT)
hdr_fbo, hdr_texture = frame_buffer_hdr(WIDTH, HEIGHT)

# model = Model("../backpack/backpack.obj")
# model = Model("../Loco/Loco.obj")
# planet_model = Model("../planet/planet.obj")
# rock_model = Model("../planet/rock.obj")
# texture = create_texture("../planet/rock1.jpg", 1)
# print("ok")

wood_texture = create_texture("./textures/wood.jpg")
# brick_diffuse_texture = create_texture("./textures/brickwall.jpg")
# brick_normal_texture = create_texture("./textures/brickwall_normal.jpg")

# depth_shader = Shader(
#     "./shaders/render_depth_cube_map.vert",
#     "./shaders/render_depth_cube_map.frag",
#     "./shaders/render_depth_cube_map.geom"
# )

shader = Shader(
    "./shaders/pre_hdr.vert",
    "./shaders/pre_hdr.frag",
)

hdr_shader = Shader(
    "./shaders/hdr.vert",
    "./shaders/hdr.frag"
)

# shadow_shader = Shader(
#     "./shaders/point_shadow.vert",
#     "./shaders/point_shadow.frag"
# )

# shadow_backpack_shader = Shader(
#     "./shaders/shadow_mapping_backpack.vs",
#     "./shaders/shadow_mapping_backpack.fs"
# )

# shader = Shader(
#     "./shaders/explode.vs",
#     "./shaders/explode.fs"
#     # "./shaders/explode.gs"
# )

# shader = Shader(
#     "./shaders/light_blinnphong.vert",
#     "./shaders/light_blinnphong.frag"
#     # "./shaders/explode.gs"
# )

# light_shader = Shader(
#     "./shaders/light.vert",
#     "./shaders/light.frag"
# )


eye = [0, 0, 50]
center = [0, 0, 0]
up = [0, 1, 0]
forward = normalize([center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]])
right = normalize(cross(forward, up))
up = normalize(cross(right, forward))
# sx = sy = sz = 1
# print(forward, right, up)
angle_x = angle_y = angle_z = 0.1
FOV = 30
aspect_ratio = WIDTH / HEIGHT
camera_speed = 3
yaw = -90
pitch = 0
sensitivity = 0.1
mode_draw = False
cutoff = 12.5
outer_cutoff = 13.5
offset = outer_cutoff - cutoff
shininess = 64
count = 1
mouse_motion_allow = False
offset_inverse_sampling = 300
exposure = 1
screen_small_pos = [0, 0]
point_light_pos = [
    0, 0, 2,
    -1.4, -1.9, 25,
     0, -1.8, 30,
     0.8, -1.7, 26
]
point_light_color = [
    200, 200, 200,
    0.1,   0,   0,
      0,   0, 0.2,
      0, 0.1,   0
]
# point_light_pos = [-3, 0, 0]
light_mode = True
gamma_correction_mode = True

light_angle = 135
light_dir = [1, -1, 0]

cube_position = [
    [4, -3.5, 0],
    [2, 3, 1],
    [-3, -1, 0],
    [-1.5, 1, 1.5],
    [-1.5, -3.5, -3]
]

cube_rotation = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [rad(60),rad(60),rad(0)]
]

cube_scale = [
    [0.5, 0.5, 0.5],
    [0.75, 0.75, 0.75],
    [0.5,0.5,0.5],
    [0.5,0.5,0.5],
    [0.75,0.75,0.75]
]

# view_light_matrix = [
#     look_at(point_light_pos[0:3], [point_light_pos[0] + 1, point_light_pos[1] + 0, point_light_pos[2] + 0], [0, 1, 0]), #right
#     look_at(point_light_pos[0:3], [point_light_pos[0] - 1, point_light_pos[1] + 0, point_light_pos[2] + 0], [0, 1, 0]), #left
#     look_at(point_light_pos[0:3], [point_light_pos[0] + 0, point_light_pos[1] + 1, point_light_pos[2] + 0], [0, 0, 1]), #top
#     look_at(point_light_pos[0:3], [point_light_pos[0] + 0, point_light_pos[1] - 1, point_light_pos[2] + 0], [0, 0, -1]), #bottom
#     look_at(point_light_pos[0:3], [point_light_pos[0] + 0, point_light_pos[1] + 0, point_light_pos[2] + 1], [0, 1, 0]), #front
#     look_at(point_light_pos[0:3], [point_light_pos[0] + 0, point_light_pos[1] + 0, point_light_pos[2] - 1], [0, 1, 0]), #back
# ]

view_light_matrix = [
    look_at(point_light_pos[0:3], [point_light_pos[0] + 1, point_light_pos[1] + 0, point_light_pos[2] + 0], [0, -1, 0]), #right
    look_at(point_light_pos[0:3], [point_light_pos[0] - 1, point_light_pos[1] + 0, point_light_pos[2] + 0], [0, -1, 0]), #left
    look_at(point_light_pos[0:3], [point_light_pos[0] + 0, point_light_pos[1] + 1, point_light_pos[2] + 0], [0, 0, 1]), #top
    look_at(point_light_pos[0:3], [point_light_pos[0] + 0, point_light_pos[1] - 1, point_light_pos[2] + 0], [0, 0, -1]), #bottom
    look_at(point_light_pos[0:3], [point_light_pos[0] + 0, point_light_pos[1] + 0, point_light_pos[2] + 1], [0, -1, 0]), #front
    look_at(point_light_pos[0:3], [point_light_pos[0] + 0, point_light_pos[1] + 0, point_light_pos[2] - 1], [0, -1, 0]), #back
]

near_light = 1
far_light = 7.5

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
                    math.cos(rad(yaw)) * math.cos(rad(pitch)),
                    math.sin(rad(pitch)),
                    math.sin(rad(yaw)) * math.cos(rad(pitch))
                ])
                right = normalize(cross(forward, [0, 1, 0]))
                up = normalize(cross(right, forward))
            # print(center)
        if event.type == MOUSEWHEEL:
            # FOV -= event.precise_y * 2
            # if FOV <= 1:
            #     FOV = 1
            # elif FOV >= 90:
            #     FOV = 89
            # print("FOV:", FOV)
            # shininess += event.precise_y * 2
            # if shininess <= 0:
            #     shininess = 0.01
            # print("shininess:", shininess)
            pass
        if event.type == KEYDOWN:
            if event.key == K_p:
                mode_draw = not mode_draw
                print("Mode draw:", "line" if mode_draw else "fill")
            if event.key == K_KP_1:
                # amount += 1000
                # prepare(amount)
                # print(amount)
                exposure += 0.1
                print("exposure:", exposure)
                pass
            if event.key == K_KP_2:
                # amount -= 1000
                # prepare(amount)
                # print(amount)
                exposure -= 0.1
                print("exposure:", exposure)
                pass
            if event.key == K_SPACE:
                mouse_motion_allow = not mouse_motion_allow
                print("Mouse_motion_allow:", mouse_motion_allow)
            # if event.key == K_RETURN:
            #     light_mode = not light_mode
                # print("Phong enabled" if light_mode else "Blinn Phong enabled")
            # if event.key == K_F1:
            #     gamma_correction_mode = not gamma_correction_mode
            #     print("Gamma_correction_mode:", gamma_correction_mode)

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
        # print(eye[2])
    if key[K_q]:
        eye = [
            eye[0] - forward[0] * camera_speed * delta_time * 2,
            eye[1] - forward[1] * camera_speed * delta_time * 2,
            eye[2] - forward[2] * camera_speed * delta_time * 2,
        ]
        # eye[1] = 0
        # print(eye[2])
    if key[K_RIGHT]:
        light_angle -= delta_time * 10
        light_dir = [-math.cos(rad(light_angle)), -math.sin(rad(light_angle)), 0]  
        print("light angle", light_angle)
    if key[K_LEFT]:
        light_angle += delta_time * 10
        light_dir = [-math.cos(rad(light_angle)), -math.sin(rad(light_angle)), 0]  
        print("light angle", light_angle)
    if key[K_UP]:
        point_light_pos[0*3+1] += camera_speed * delta_time
        point_light_pos[1*3+1] += camera_speed * delta_time
        point_light_pos[2*3+1] += camera_speed * delta_time
        point_light_pos[3*3+1] += camera_speed * delta_time
        print(point_light_pos[0:3])
    if key[K_DOWN]:
        point_light_pos[0*3+1] -= camera_speed * delta_time
        point_light_pos[1*3+1] -= camera_speed * delta_time
        point_light_pos[2*3+1] -= camera_speed * delta_time
        point_light_pos[3*3+1] -= camera_speed * delta_time
        print(point_light_pos[0:3])
    if key[K_i]:
        near_light += 2 * delta_time
        print("near:", near_light)
    if key[K_k]:
        near_light -= 2 * delta_time
        print("near:", near_light)
    if key[K_j]:
        far_light -= 2 * delta_time
        print("far:", far_light)
    if key[K_l]:
        far_light += 2 * delta_time
        print("far:", far_light)
    # if key[K_KP_1]:
    #     exposure -= 0.1
    #     print("exposure:", exposure)
    # if key[K_KP_2]:
    #     exposure += 0.1
    #     print("exposure:", exposure)

    # u_model = get_model_matrix(rotate=rotate(rad(-55), 0, 0))
    # 1
    # glClearColor(0.1, 0.1, 0.1, 1)
    # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
    if mode_draw:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    else:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    # if gamma_correction_mode:
    #     glEnable(GL_FRAMEBUFFER_SRGB)
    # else:
    #     glDisable(GL_FRAMEBUFFER_SRGB)

    # glDepthMask(GL_TRUE)
    # glDepthFunc(GL_LEQUAL)

    # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    u_view = look_at(eye, [eye[0] + forward[0], eye[1] + forward[1], eye[2] + forward[2]], up)
    # u_view_skybox = look_at_no_translate(eye, [eye[0] + forward[0], eye[1] + forward[1], eye[2] + forward[2]], up)
    u_proj = perspective(rad(FOV), aspect_ratio, 0.1, 1000)

    # u_light_space_matrix = ortho(-10, 10, -10, 10, near_light, far_light) @ look_at(
    #     point_light_pos[0:3], 
    #     # (0, 0, 0), 
    #     (point_light_pos[0] + light_dir[0], point_light_pos[1] + light_dir[1], point_light_pos[2] + light_dir[2]), 
    #     (0, 1, 0)
    # )
    
    # far_light = 25.0
    # u_proj_light_matrix = perspective(rad(90), DEPTH_WIDTH / DEPTH_HEIGHT, 1.0, far_light)

    # glBindFramebuffer(GL_FRAMEBUFFER, depth_fbo)

    # glViewport(0, 0, DEPTH_WIDTH, DEPTH_HEIGHT)
    # glEnable(GL_DEPTH_TEST)
    # glCullFace(GL_FRONT)
    # # glClearColor(0.1, 0.1, 0.1, 1)
    # glClear(GL_DEPTH_BUFFER_BIT)
    # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # depth_shader.use()
    # for i in range(len(view_light_matrix)):
    #     depth_shader.setMatrix4(f"uViewLight[{i}]", view_light_matrix[i])
    # depth_shader.setMatrix4("uProjLight", u_proj_light_matrix)
    # depth_shader.setFloat3("pointLightPos", *point_light_pos[0:3])
    # depth_shader.setFloat("farLight", far_light)
 
    # room_scalar = 5
    # u_model = get_model_matrix(scale=scale(room_scalar, room_scalar, room_scalar))
    # depth_shader.setMatrix4("uModel", u_model)
    # glBindVertexArray(cube_vao)
    # # glCullFace(GL_BACK)
    # glDrawArrays(GL_TRIANGLES, 0, 36)

    # # glCullFace(GL_FRONT)
    # for i in range(len(cube_position)):
    #     u_model = get_model_matrix(
    #         translate=translate(*cube_position[i]),
    #         rotate=rotate(*cube_rotation[i]),
    #         scale=scale(*cube_scale[i])
    #     )
    #     depth_shader.setMatrix4("uModel", u_model)
    #     glBindVertexArray(cube_vao)
    #     glDrawArrays(GL_TRIANGLES, 0, 36)

    glBindFramebuffer(GL_FRAMEBUFFER, hdr_fbo)
    glEnable(GL_DEPTH_TEST)
    glViewport(0, 0, WIDTH, HEIGHT)
    # glClearColor(0.1, 0.1, 0.1, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    shader.use()
    shader.setMatrix4("uView", u_view)
    shader.setMatrix4("uProj", u_proj)
    shader.setBool("uInverseNormal", True)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, wood_texture)
    shader.setInt("texture1", 0)
    shader.setFloat3("ViewPos", *eye)
    shader.setFloat("shininess", 32)

    for i in range(len(point_light_pos) // 3):
        shader.setFloat3(f"light.pos[{i}]", *point_light_pos[i*3:i*3+3])
        shader.setFloat3(f"light.color[{i}]", *point_light_color[i*3:i*3+3])

    u_model = get_model_matrix(
        translate=translate(0, 0, 25),
        # rotate=rotate(0, rad(45), 0),
        scale=scale(2.5, 2.5, 27.5)
        # scale=scale(27.5, , 27.5)
    )
    shader.setMatrix4("uModel", u_model)
    glBindVertexArray(cube_vao)
    glCullFace(GL_FRONT)
    glDrawArrays(GL_TRIANGLES, 0, 36)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glDisable(GL_DEPTH_TEST)
    glClear(GL_COLOR_BUFFER_BIT)

    hdr_shader.use()
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, hdr_texture)
    hdr_shader.setInt("hdrTexture", 1)
    hdr_shader.setFloat("exposure", exposure)
    glCullFace(GL_BACK)
    glBindVertexArray(screen_vao)
    glDrawArrays(GL_TRIANGLES, 0, 6)

    # u_model = get_model_matrix(scale=scale(room_scalar, room_scalar, room_scalar))
    # shadow_shader.setMatrix4("uModel", u_model)
    # glBindVertexArray(cube_vao)
    # # glDisable(GL_CULL_FACE)
    # glCullFace(GL_FRONT)
    # # glCullFace(GL_NONE)
    # glDrawArrays(GL_TRIANGLES, 0, 36)

    # # glEnable(GL_CULL_FACE)
    # glCullFace(GL_BACK)
    # for i in range(len(cube_position)):
    #     u_model = get_model_matrix(
    #         translate=translate(*cube_position[i]),
    #         rotate=rotate(*cube_rotation[i]),
    #         scale=scale(*cube_scale[i])
    #     )
    #     shadow_shader.setMatrix4("uModel", u_model)
    #     glBindVertexArray(cube_vao)
    #     glDrawArrays(GL_TRIANGLES, 0, 36)

    # shadow_shader.use()
    # # u_proj = ortho(-1, 1, -1, 1, 0, 1)
    # shadow_shader.setMatrix4("uProj", u_proj)
    # shadow_shader.setMatrix4("uView", u_view)
    # glActiveTexture(GL_TEXTURE0 + 7)
    # glBindTexture(GL_TEXTURE_2D, depth_map)
    # shadow_shader.setInt("depthMap", 7)
    # glActiveTexture(GL_TEXTURE0 + 4)
    # glBindTexture(GL_TEXTURE_2D, 4)
    # shadow_shader.setInt("diffuseMap", wood_texture)
    # shadow_shader.setMatrix4("uLightSpace", u_light_space_matrix)
    # shadow_shader.setFloat3("uLightColor", 1, 1, 1)
    # shadow_shader.setFloat3("uLightDir", *light_dir)
    # shadow_shader.setFloat3("uViewPos", *eye)
    # glBindVertexArray(screen_vao)

    # u_model = get_model_matrix()
    # shadow_shader.setMatrix4("uModel", u_model)
    # glBindVertexArray(plane_vao)
    # glCullFace(GL_FRONT)
    # glDrawArrays(GL_TRIANGLES, 0, 6)

    # glCullFace(GL_BACK)
    # for i in range(len(cube_position)):
    #     u_model = get_model_matrix(
    #         translate=translate(*cube_position[i]),
    #         rotate=rotate(*cube_rotation[i]),
    #         scale=scale(0.5, 0.5, 0.5))
    #     shadow_shader.setMatrix4("uModel", u_model)
    #     glBindVertexArray(cube_vao)
    #     glDrawArrays(GL_TRIANGLES, 0, 36)

    # shadow_backpack_shader.use()
    # # u_proj = ortho(-1, 1, -1, 1, 0, 1)
    # shadow_backpack_shader.setMatrix4("uProj", u_proj)
    # shadow_backpack_shader.setMatrix4("uView", u_view)
    # glActiveTexture(GL_TEXTURE0 + 7)
    # glBindTexture(GL_TEXTURE_2D, depth_map)
    # shadow_backpack_shader.setInt("depthMap", depth_map)
    # # shadow_backpack_shader.setInt("diffuseMap", wood_texture)
    # shadow_backpack_shader.setMatrix4("uLightSpace", u_light_space_matrix)
    # shadow_backpack_shader.setFloat3("uLightColor", 1, 1, 1)
    # shadow_backpack_shader.setFloat3("uLightDir", *light_dir)
    # shadow_backpack_shader.setFloat3("uViewPos", *eye)
    # u_model = get_model_matrix(translate(-3, 1, 0), scale=scale(0.7, 0.7, 0.7))
    # shadow_backpack_shader.setMatrix4("uModel", u_model)
    # model.draw(shadow_backpack_shader)
    
    # light_shader.use()
    # light_shader.setMatrix4("uView", u_view)
    # light_shader.setMatrix4("uProj", u_proj)
    # u_model = get_model_matrix(translate(*point_light_pos[0:3]), scale=scale(0.03, 0.03, 0.03))
    # light_shader.setMatrix4("uModel", u_model)
    # glBindVertexArray(light_vao)
    # glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)

    # shader.use()
    # u_model = get_model_matrix(
    #     translate=translate(0, 0, 0),
    #     rotate=rotate(0, 0, 0), 
    #     scale=scale(5, 1, 5)
    # )
    # shader.setMatrix4("u_model", u_model)
    # shader.setMatrix4("u_view", u_view)
    # shader.setMatrix4("u_proj", u_proj)

    # shader.setFloat3("viewPos", *eye)
    # shader.setInt("texture1", texture)
    # shader.setFloat("shininess", shininess)


    # shader.setFloat3v("light.pos", len(point_light_pos) // 3, point_light_pos)
    # shader.setFloat3v("light.ambient", len(point_light_color) // 3, point_light_color)
    # shader.setFloat3v("light.diffuse", len(point_light_color) // 3, point_light_color)
    # shader.setFloat3v("light.specular", len(point_light_color) // 3, point_light_color)
    # shader.setFloat("light.constant", 1)
    # shader.setFloat("light.linear", 0.22)
    # shader.setFloat("light.quadratic", 0.2)

    # shader.setFloat("light.constant", 1)
    # shader.setFloat("light.linear", 0.0014)
    # shader.setFloat("light.quadratic", 0.000007)
    # glBindVertexArray(plane_vao)

    # glViewport(0, 0, WIDTH // 2, HEIGHT)
    # shader.setInt("gammaCorrectionMode", not gamma_correction_mode)
    # glDrawArrays(GL_TRIANGLES, 0, len(plane_vertices) // 6)

    # glViewport(WIDTH // 2, 0, WIDTH // 2, HEIGHT)
    # shader.setInt("gammaCorrectionMode", gamma_correction_mode)
    # glDrawArrays(GL_TRIANGLES, 0, len(plane_vertices) // 6)

    # light_shader.use()
    # light_shader.setMatrix4("u_view", u_view)
    # light_shader.setMatrix4("u_proj", u_proj)
    # glBindVertexArray(light_vao)
    # for i in range(len(point_light_pos) // 3):
    # u_model = get_model_matrix(
    #     translate=translate(*point_light_pos[0:3]),
    #     rotate=rotate(0, 0, 0), 
    #     scale=scale(0.2, 0.2, 0.2)
    # )
    #     light_shader.setMatrix4("u_model", u_model)
    # light_shader.setMatrix4("u_model", u_model)
        # glViewport(0, 0, WIDTH // 2, HEIGHT)
    # glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)

        # glViewport(WIDTH // 2, 0, WIDTH // 2, HEIGHT)
        # glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)

    pygame.display.flip()
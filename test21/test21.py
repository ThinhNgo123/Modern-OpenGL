import pygame, sys, math
from pygame.locals import *
from OpenGL.GL import *
import numpy as np
import ctypes 
import glm
from model import Model
from shader import Shader

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
        [1,           0,            0, 0],
        [0, math.cos(a), -math.sin(a), 0],
        [0, math.sin(a),  math.cos(a), 0],
        [0,           0,            0, 1]
    ])

def rotate_y(a):
    return np.array([
        [ math.cos(a), 0, math.sin(a), 0],
        [           0, 1,           0, 0],
        [-math.sin(a), 0, math.cos(a), 0],
        [           0, 0,           0, 1]
    ])

def rotate_z(a):
    return np.array([
        [math.cos(a), -math.sin(a), 0, 0],
        [math.sin(a),  math.cos(a), 0, 0],
        [          0,            0, 1, 0],
        [          0,            0, 0, 1]
    ])

def rotate(alpha, beta, gamma):
    return rotate_x(alpha) @ rotate_y(beta) @ rotate_z(gamma)

def ortho(left, right, bottom, top, near, far):
    return np.array([
        [2 / (right - left),                  0,                0, - (right + left) / (right - left)],
        [                 0, 2 / (top - bottom),                0, - (top + bottom) / (top - bottom)],
        [                 0,                  0, 2 / (near - far),     - (far + near) / (far - near)],
        [                 0,                  0,                0,                                 1]
    ])

def frustum(l, r, b, t, n, f):
    assert (0 < n < f), "Near, far < 0 or near > far"
    return np.array([
        [(2 * n) / (r - l),                 0,  (r + l) / (r - l),                      0],
        [                0, (2 * n) / (t - b),  (t + b) / (t - b),                      0],
        [                0,                 0, -(f + n) / (f - n), (-2 * f * n) / (f - n)],
        [                0,                 0,                 -1,                      0]
    ])

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
    ]) @ \
    np.array([
        [1, 0, 0, -eye[0]],
        [0, 1, 0, -eye[1]],
        [0, 0, 1, -eye[2]],
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

    return vao, vbo

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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, 
        image.get_width(),
        image.get_height(),
        0, GL_RGB, GL_UNSIGNED_BYTE, pygame.image.tostring(image, "RGB"))
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

WIDTH, HEIGTH, FPS = 1200, 800, 120
pygame.init()
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 6)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
# pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
# pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
# pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 2)
win = pygame.display.set_mode((WIDTH, HEIGTH), flags=OPENGL | DOUBLEBUF)
# glEnable(GL_MULTISAMPLE)
glEnable(GL_DEPTH_TEST)
# glFrontFace(GL_CCW)
# glFrontFace(GL_CW)
# glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
# glEnable(GL_CULL_FACE)
# glCullFace(GL_FRONT)
# glCullFace(GL_BACK)
# glLineWidth(2)
# pygame.event.set_grab(True)
# pygame.mouse.set_visible(False)
# pygame.mouse.set_pos(WIDTH / 2, HEIGTH / 2)
print("Version:", glGetString(GL_VERSION))
print("Shader version:", glGetString(GL_SHADING_LANGUAGE_VERSION))
clock = pygame.time.Clock()
current_frame = last_frame = 0
# vertices, faces = load_obj("../bugatti/bugatti.obj")
# vertices, faces = load_obj("../../learn_python/Software_3D_engine-main/resources/t_34_obj.obj")
# vertices, faces = load_obj("../MI28.obj")

point_light_position = [
    [0.7,  0.2,   2],
    [2.3, -3.3,  -4],
    [ -4,    2, -12],
    [  0,    0,  -3]
]

light_pos = [1.2, 1, 2]

model = Model("../Loco/Loco.obj")
# model = Model("./xetang.obj")
shader = Shader(
    # "./shaders/backpack.vert",
    # "./shaders/backpack.frag"
    "./shaders/model.vert",
    "./shaders/model.frag"
)

diffuse = create_texture("../backpack/diffuse.jpg", 0)
shader.use()

light_vao = create_light_vao()
light_shader = create_shader(
    "./shaders/light.vert",
    "./shaders/light.frag"
)
light_model_location = glGetUniformLocation(light_shader, "u_model")
light_view_location = glGetUniformLocation(light_shader, "u_view")
light_proj_location = glGetUniformLocation(light_shader, "u_proj")
light_position = [5, 5, 5]
# light_direction = [-0.2, -1, -0.3]
axis_world_vao = axis()
# axis_local_vao = axis(1)

# shader.use()
# glUseProgram(program)
# model_location = glGetUniformLocation(program, "u_model")
# view_location = glGetUniformLocation(program, "u_view")
# proj_location = glGetUniformLocation(program, "u_proj")

# eye_position_location = glGetUniformLocation(program, "viewPos")

# diffuse_location = glGetUniformLocation(program, "material.diffuse")
# specular_location = glGetUniformLocation(program, "material.specular")
# # emission_location = glGetUniformLocation(program, "material.emission")
# shininess_location = glGetUniformLocation(program, "material.shininess")

# direction_dir_light_location = glGetUniformLocation(program, "dirLight.direction")
# ambient_dir_light_location = glGetUniformLocation(program, "dirLight.ambient")
# diffuse_dir_light_location = glGetUniformLocation(program, "dirLight.diffuse")
# specular_dir_light_location = glGetUniformLocation(program, "dirLight.specular")

# point_light_location = []

# for i in range(len(point_light_position)):
#     point_light_location.append({
#         "position": glGetUniformLocation(program, f"pointLight[{i}].position"),
#         "ambient": glGetUniformLocation(program, f"pointLight[{i}].ambient"),
#         "diffuse": glGetUniformLocation(program, f"pointLight[{i}].diffuse"),
#         "specular": glGetUniformLocation(program, f"pointLight[{i}].specular"),
#         "constant": glGetUniformLocation(program, f"pointLight[{i}].constant"),
#         "linear": glGetUniformLocation(program, f"pointLight[{i}].linear"),
#         "quadratic": glGetUniformLocation(program, f"pointLight[{i}].quadratic")
#     })

# position_spot_light_location = glGetUniformLocation(program, "spotLight.position")
# direction_spot_light_location = glGetUniformLocation(program, "spotLight.direction")
# cutoff_spot_light_location = glGetUniformLocation(program, "spotLight.cutOff")
# outer_cutoff_spot_light_location = glGetUniformLocation(program, "spotLight.outerCutOff")
# ambient_spot_light_location = glGetUniformLocation(program, "spotLight.ambient")
# diffuse_spot_light_location = glGetUniformLocation(program, "spotLight.diffuse")
# specular_spot_light_location = glGetUniformLocation(program, "spotLight.specular")

# constant_spot_light_location = glGetUniformLocation(program, "spotLight.constant")
# linear_spot_light_location = glGetUniformLocation(program, "spotLight.linear")
# quadratic_spot_light_location = glGetUniformLocation(program, "spotLight.quadratic")


# glUseProgram(light_program)
# model1_location = glGetUniformLocation(light_program, "u_model")
# view1_location = glGetUniformLocation(light_program, "u_view")
# proj1_location = glGetUniformLocation(light_program, "u_proj")
# light_color_location = glGetUniformLocation(light_program, "lightColor")
# u_proj = ortho(-1, 1, -1, 1, 1, -1)
# print(u_proj)
# print(glm.ortho(-1, 1, -1, 1, -1, 1))

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
model_draw = False
cutoff = 12.5
outer_cutoff = 13.5
offset = outer_cutoff - cutoff
shininess = 10
# glEnable(GL_SCISSOR_TEST)

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
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION:
            rel_x, rel_y = pygame.mouse.get_rel()
            # print(rel_x, rel_y)
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
            print("shininess:", shininess)
        if event.type == KEYDOWN:
            if event.key == K_p:
                model_draw = not model_draw

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

    # if key[K_UP]:
    #     light_pos[1] += 1 * delta_time
    # if key[K_DOWN]:
    #     light_pos[1] -= 1 * delta_time
    # if key[K_RIGHT]:
    #     cutoff += 0.05
    #     outer_cutoff += 0.05
    #     print(cutoff)
    #     pass
    # if key[K_LEFT]:
    #     cutoff -= 0.05
    #     outer_cutoff -= 0.05
    #     if cutoff <= 0:
    #         cutoff = 0
    #         outer_cutoff = offset
    #     print(cutoff)

    if key[K_UP]:
        light_position[1] += 3 * delta_time
        print(light_position)
    if key[K_DOWN]:
        light_position[1] -= 3 * delta_time
        print(light_position)
    if key[K_LEFT]:
        light_position[0] -= 3 * delta_time
        print(light_position)
    if key[K_RIGHT]:
        light_position[0] += 3 * delta_time
        print(light_position)
    if key[K_1]:
        light_position[2] += 3 * delta_time
        print(light_position)
    if key[K_2]:
        light_position[2] -= 3 * delta_time
        print(light_position)


    # u_model = get_model_matrix(rotate=rotate(rad(-55), 0, 0))
    # 1
    # glViewport(0, HEIGTH // 2, WIDTH // 2, HEIGTH // 2)
    # glScissor(0, HEIGTH // 2, WIDTH // 2, HEIGTH // 2)
    glClearColor(0.2, 0.2, 0.2, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    if model_draw:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    else:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    u_view = look_at(eye, [eye[0] + forward[0], eye[1] + forward[1], eye[2] + forward[2]], up)
    # print(eye)
    # print(eye[2])
    # u_view = glm.value_ptr(glm.lookAt(glm.vec3(0, 0, radius), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0)))
    u_proj = perspective(rad(FOV), aspect_ratio, 0.1, 100)

    # u_model = get_model_matrix()
    u_model = get_model_matrix(scale=scale(0.01, 0.01, 0.01))
    # u_model = get_model_matrix(scale=scale(0.05, 0.05, 0.05))

    shader.use()
    shader.setMatrix4("u_view", u_view)
    shader.setMatrix4("u_proj", u_proj)
    shader.setMatrix4("u_model", u_model)
    # shader.setInt("diffuse", diffuse)
    shader.setFloat3("viewDir", *eye)
    shader.setFloat3("dirLight.direction", -light_position[0], -light_position[1], -light_position[2])
    shader.setFloat3("dirLight.ambient", 0.2, 0.2, 0.2)
    shader.setFloat3("dirLight.diffuse", 0.5, 0.5, 0.5)
    shader.setFloat3("dirLight.specular", 1, 1, 1)

    shader.setFloat3("material.ambient", 0.5882, 0.5882, 0.5882)
    shader.setFloat3("material.diffuse", 0.5882, 0.5882, 0.5882)
    shader.setFloat3("material.specular", 1, 1, 1)
    shader.setFloat("material.shininess", shininess)

    model.draw(shader)

    glUseProgram(light_shader)
    glUniformMatrix4fv(light_view_location, 1, GL_TRUE, u_view)
    glUniformMatrix4fv(light_proj_location, 1, GL_TRUE, u_proj)
    u_model = get_model_matrix(
        translate=translate(*light_position),
        scale=scale(0.2, 0.2, 0.2)
        # rotate=rotate(angle_x * i, angle_y * i, angle_z)
    )
    glUniformMatrix4fv(light_model_location, 1, GL_TRUE, u_model)
    glBindVertexArray(light_vao)
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT , None)

    # u_model = get_model_matrix(
    #         translate=translate(0, 0, 0),
    #         # scale=scale(2, 2, 2)
    # )
    # glUniformMatrix4fv(light_model_location, 1, GL_TRUE, u_model)
    # glBindVertexArray(axis_world_vao)
    # glDrawArrays(GL_LINES, 0, 6)

    # glUseProgram(program)
    # glUniformMatrix4fv(view_location, 1, GL_TRUE, u_view)
    # glUniformMatrix4fv(proj_location, 1, GL_TRUE, u_proj)

    # glUniform3f(eye_position_location, *eye)

    # glUniform1i(diffuse_location, diffuse_texture)
    # glUniform1i(specular_location, specular_texture)
    # # glUniform1i(emission_location, emission_texture)
    # glUniform1f(shininess_location, 25)

    # glUniform3f(direction_dir_light_location, -0.2, -1, -0.3)
    # glUniform3f(ambient_dir_light_location, 0.05, 0.05, 0.05)
    # glUniform3f(diffuse_dir_light_location, 0.4, 0.4, 0.4)
    # glUniform3f(specular_dir_light_location, 0.5, 0.5, 0.5)

    # for i in range(len(point_light_position)):
    #     glUniform3f(point_light_location[i]["position"], *point_light_position[i])
    #     # print("forward", forward)
    #     glUniform3f(point_light_location[i]["ambient"], 0.01, 0, 0)
    #     glUniform3f(point_light_location[i]["diffuse"], 0.5, 0, 0)
    #     glUniform3f(point_light_location[i]["specular"], 1, 1, 1)

    #     glUniform1f(point_light_location[i]["constant"], 1)
    #     glUniform1f(point_light_location[i]["linear"], 0.022)
    #     glUniform1f(point_light_location[i]["quadratic"], 0.0019)

    # glUniform3f(position_spot_light_location, *eye)
    # glUniform3f(direction_spot_light_location, *forward)
    # # print("forward", forward)
    # glUniform1f(cutoff_spot_light_location, math.cos(rad(cutoff)))
    # glUniform1f(outer_cutoff_spot_light_location, math.cos(rad(outer_cutoff)))
    # glUniform3f(ambient_spot_light_location, 0.1, 0.1, 0.1)
    # glUniform3f(diffuse_spot_light_location, 0.8, 0.8, 0.8)
    # glUniform3f(specular_spot_light_location, 1, 1, 1)
    # # glUniform3f(ambient_spot_light_location, am, am, am)
    # # glUniform3f(diffuse_spot_light_location, am, am, am)
    # # glUniform3f(specular_spot_light_location, am, am, am)

    # glUniform1f(constant_spot_light_location, 1)
    # glUniform1f(linear_spot_light_location, 0.022)
    # glUniform1f(quadratic_spot_light_location, 0.0019)

    # for index, position in enumerate(cube_positions):
    # # for x in range(10):
    # #     for z in range(10):
    #     u_model = get_model_matrix(
    #         # translate=translate(0, 0, 0),
    #         translate=translate(*position),
    #         # translate=translate(x, 0, -z),
    #         # scale=scale(2, 2, 2)
    #         # rotate=rotate(angle_x * index, angle_y * index, angle_z * index)
    #     )
    #     glUniformMatrix4fv(model_location, 1, GL_TRUE, u_model)
    #     glBindVertexArray(vao)
    #     glDrawArrays(GL_TRIANGLES, 0, 36)
    #     # glBindVertexArray(axis_local_vao)
    #     # glDrawArrays(GL_LINES, 0, 6)

    # u_model = get_model_matrix(
    #         translate=translate(0, 0, 0),
    #         # scale=scale(2, 2, 2)
    # )
    # glUniformMatrix4fv(model_location, 1, GL_TRUE, u_model)
    # glBindVertexArray(axis_world_vao)
    # glDrawArrays(GL_LINES, 0, 6)

    # glUseProgram(light_program)
    # glUniformMatrix4fv(view1_location, 1, GL_TRUE, u_view)
    # glUniformMatrix4fv(proj1_location, 1, GL_TRUE, u_proj)
    # glUniform3f(light_color_location, 1, 0, 0)
    # for i in range(len(point_light_position)):
    #     u_model = get_model_matrix(
    #         translate=translate(*point_light_position[i]),
    #         scale=scale(0.2, 0.2, 0.2)
    #         # rotate=rotate(angle_x * i, angle_y * i, angle_z)
    #     )
    #     glUniformMatrix4fv(model1_location, 1, GL_TRUE, u_model)
    #     glBindVertexArray(light_vao)
    #     glDrawArrays(GL_TRIANGLES, 0, 36)

    # glDrawElements(GL_TRIANGLES, len_faces, GL_UNSIGNED_INT, None)

    # angle_x += 0.007
    # angle_y += 0.007
    # angle_z += 0.01

    pygame.display.flip()
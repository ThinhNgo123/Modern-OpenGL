import pygame
from OpenGL.GL import *
from pyassimp import load, postprocess
from PIL import Image
from typing import List, Dict
from mesh import Mesh, Vertex, Texture
from shader import Shader

class Model:
    def __init__(self, path: str) -> None:
        self.meshes: List[Mesh] = []
        self.textures_loaded: Dict[str, Texture] = {}
        self.directory: str = ""

        self.load_model(path)
        
        # mesh = self.meshes[0]
        # print(mesh.vertices[0].position)
        # print(mesh.vertices[0].normal)
        # print(mesh.vertices[0].tex_coords)
        # for mesh in self.meshes:
        #     print(mesh.textures[0].type)
        #     print(mesh.textures[1].type)
        #     print()


    def load_model(self, path: str):
        with load(
            path, 
            processing=postprocess.aiProcess_Triangulate | postprocess.aiProcess_FlipUVs
            # processing=postprocess.aiProcess_Triangulate
        ) as scene:

            self.directory = path[:path.rfind("/")+1]
            self.processNode(scene.rootnode, scene)

    def processNode(self, node, scene):
        for mesh in node.meshes:
            self.meshes.append(self.processMesh(mesh, scene))
        for children in node.children:
            self.processNode(children, scene)

    def processMesh(self, mesh, scene):
        vertices: List[Vertex] = []
        indices: List[int] = []
        textures: List[Texture] = []
        for index in range(len(mesh.vertices)):
            position = [
                mesh.vertices[index][0], 
                mesh.vertices[index][1], 
                mesh.vertices[index][2]
            ]
            normal = [
                mesh.normals[index][0],
                mesh.normals[index][1],
                mesh.normals[index][2]
            ]
            if len(mesh.texturecoords) == 1:
                tex_coord = [
                    mesh.texturecoords[0][index][0],
                    mesh.texturecoords[0][index][1]
                ]
            else:
                tex_coord = [0, 0]
            vertices.append(Vertex(position, normal, tex_coord))
        
        for face in mesh.faces:
            indices.extend([face[0], face[1], face[2]])

        # print(indices)
        # print()

        textures.extend(self.load_material_textures(mesh.material, "texture_diffuse"))
        textures.extend(self.load_material_textures(mesh.material, "texture_specular"))

        return Mesh(vertices, indices, textures)

    def texture_from_file(self, path: str):
        # image = pygame.image.load(path)
        image = Image.open(path)
        # image = self.load_image("./textures/wall.jpg")
        # image = pygame.transform.flip(image, flip_x=False, flip_y=True)
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        texture_id = glGenTextures(1)
        
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        
        if path.split(".")[-1] == "png":
            format = GL_RGBA
        else:
            format = GL_RGB
        # print(format)
        glTexImage2D(
            GL_TEXTURE_2D, 0, format, 
            image.width,
            image.height,
            0, format, GL_UNSIGNED_BYTE, image.tobytes())
        glGenerateMipmap(GL_TEXTURE_2D)

        return texture_id

    def load_material_textures(self, material, type_name: str):
        textures = []
        type = type_name.split("_")[1]
        # print("type", type)
        for key, value in material.properties.items():
            # print("items:", key, value)
            if key != "file":# or value.split(".")[0] != type:
                continue
            # print("ok")
            texture = self.textures_loaded.get(value, None)
            if texture:
                textures.append(texture)
                continue
            texture = Texture(
                id=self.texture_from_file(self.directory + value),
                type=type_name,
                path=value
            )
            self.textures_loaded[value] = texture
            textures.append(texture)
        # print(textures)
        return textures

    def draw(self, shader: Shader):
        for mesh in self.meshes:
            mesh.draw(shader)
        # self.meshes[50].draw(shader)

if __name__ == "__main__":
    model = Model("../backpack/backpack.obj")

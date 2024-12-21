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
        self.directory: str = path[:path.rfind("/")+1]
        self.materials = {}

        self.load_mtl_file(path)
        self.load_model(path)
        
        # mesh = self.meshes[0]
        # print(mesh.vertices[0].position)
        # print(mesh.vertices[0].normal)
        # print(mesh.vertices[0].tex_coords)
        # for mesh in self.meshes:
        #     print(mesh.textures[0].type)
        #     print(mesh.textures[1].type)
        #     print()

    def load_mtl_file(self, path: str):
        # print(path[path.rfind("/")+1:].split(".")[0])
        with open(self.directory + \
                  path[path.rfind("/")+1:].split(".")[0] + ".mtl"
        ) as file:
            name = ""
            for line in file:
                # print(line)
                if line[0] == "#" or line.strip("\n") == "":
                    continue
                # print(line.strip("\n").strip("\t").split(" ", maxsplit=1))
                word1, word2 = line.strip("\n").strip("\t").split(" ", maxsplit=1)
                if word1 == "newmtl":
                    name = word2
                    self.materials[word2] = {}
                elif word1 in ("Tf", "Ka", "Kd", "Ks", "Ke"):
                    self.materials[name][word1] = [float(value) for value in word2.strip(" ").split(" ")]
                elif word1 in ("Ns", "Ni", "d", "Tr", "illum", "map_Ka", "map_Kd", "map_Ks", "map_d"):
                    self.materials[name][word1] = word2

    def load_model(self, path: str):
        with load(
            path, 
            processing=postprocess.aiProcess_Triangulate | postprocess.aiProcess_FlipUVs
            # processing=postprocess.aiProcess_Triangulate
        ) as scene:

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
        image = Image.open(path).convert("RGB")
        # image = self.load_image("./textures/wall.jpg")
        # image = pygame.transform.flip(image, flip_x=False, flip_y=True)
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        # image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        texture_id = glGenTextures(1)
        # print("texture id:", texture_id)
        
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB, 
            image.width,
            image.height,
            0, GL_RGB, GL_UNSIGNED_BYTE, image.tobytes())
        glGenerateMipmap(GL_TEXTURE_2D)

        return texture_id

    def load_material_textures(self, material, type_name: str):
        textures = []
        material = self.materials[material.properties["name"]]
            # print("ok")
        if type_name == "texture_diffuse" and \
            material.get("map_Kd", None):
            
            texture = self.textures_loaded.get(material["map_Kd"], None)
            if texture:
                textures.append(texture)
            else:
                texture = Texture(
                    id=self.texture_from_file(self.directory + material["map_Kd"]),
                    type=type_name,
                    path=material["map_Kd"]
                )
                self.textures_loaded[material["map_Kd"]] = texture
                textures.append(texture)
        elif type_name == "texture_specular" and \
            material.get("map_Ks", None):
            
            texture = self.textures_loaded.get(material["map_Ks"], None)
            if texture:
                textures.append(texture)
            else:
                texture = Texture(
                    id=self.texture_from_file(self.directory + material["map_Ks"]),
                    type=type_name,
                    path=material["map_Ks"]
                )
                self.textures_loaded[material["map_Ks"]] = texture
                textures.append(texture)
        # for texture in textures:
        #     print(texture.__dict__)
        return textures

    def draw(self, shader: Shader):
        for mesh in self.meshes:
            mesh.draw(shader)
        # self.meshes[0].draw(shader)

if __name__ == "__main__":
    model = Model("../backpack/backpack.obj")

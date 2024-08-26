#version 460 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

out vec3 FragPos;
out vec3 Normal;
out vec2 TextureCoord;

void main()
{
    
    gl_Position = u_proj * u_view * u_model * vec4(aPos, 1);
    // gl_Position = u_model * vec4(vertexPos, 1);
    // gl_Position = u_view * u_model * vec4(vertexPos, 1);
    // coord = vertexCoord;
    FragPos = vec3(u_model * vec4(aPos, 1));
    // Normal = vertexNormal;
    Normal = mat3(transpose(inverse(u_model))) * aNormal;
    TextureCoord = aTexCoord;
}
#version 460 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

// out vec2 TexCoord;
out vec3 FragPos;
out vec3 Normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

void main()
{

    gl_Position = u_proj * u_view * u_model * vec4(aPos, 1);
    // gl_Position = u_model * vec4(vertexPos, 1);
    // gl_Position = u_view * u_model * vec4(vertexPos, 1);
    // TexCoord = aTexCoord;
    FragPos = (u_model * vec4(aPos, 1)).xyz;
    Normal = mat3(transpose(inverse(u_model))) * aNormal; 
}
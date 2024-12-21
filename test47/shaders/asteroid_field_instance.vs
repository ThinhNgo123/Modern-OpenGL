#version 460 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in mat4 aModelMatrix;

// out vec3 Normal;
out vec2 TexCoord;

// uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

void main()
{
    gl_Position = u_proj * u_view * aModelMatrix * vec4(aPos, 1);
    TexCoord = aTexCoord;
}
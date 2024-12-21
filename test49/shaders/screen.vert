#version 460 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aTexCoord;

out vec3 TexCoord;

uniform mat4 u_model;

void main()
{
    gl_Position = u_model * vec4(aPos, 1);
    TexCoord = aTexCoord;
}
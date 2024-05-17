#version 460 core

layout (location = 0) in vec3 vertexPos;

out vec4 pos;

uniform mat4 u_model;

void main()
{
    gl_Position = u_model * vec4(vertexPos, 1);
    pos = gl_Position;
}
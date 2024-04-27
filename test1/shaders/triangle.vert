#version 460 core

layout (location = 0) in vec4 vertexPos;

out vec4 pos;

uniform mat4 translate;
uniform mat4 scale;
uniform mat4 rotate_y;

void main()
{
    gl_Position = vertexPos * translate * scale * rotate_y;
    pos = gl_Position;
}
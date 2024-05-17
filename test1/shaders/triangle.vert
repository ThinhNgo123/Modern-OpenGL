#version 460 core

layout (location = 0) in vec3 vertexPos;
layout (location = 1) in vec2 texPos;

out vec4 pos;
out vec2 texCoord;

uniform mat4 translate;
uniform mat4 scale;
uniform mat4 rotate_y;

void main()
{
    gl_Position = vec4(vertexPos, 1) * translate * scale * rotate_y;
    pos = gl_Position;
    texCoord = texPos;
}
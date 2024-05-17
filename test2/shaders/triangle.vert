#version 460 core

layout (location = 0) in vec3 vertexPos;

out vec4 pos;

uniform mat4 translate;
uniform mat4 scale;
uniform mat4 rotate;

void main()
{
    gl_Position = vec4(vertexPos, 1) * rotate * translate * scale;
    pos = gl_Position;
}
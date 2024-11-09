#version 460 core

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;

out vec3 Color;

uniform vec2 translation[100];

void main()
{
    int count = 10000;
    int devide = count / 100;
    gl_Position = vec4(aPos + translation[gl_InstanceID / devide], 0, 1);
    Color = aColor;
}
#version 460 core

layout (location = 0) in vec3 vertexPos;
layout (location = 1) in vec2 vertexCoord;

out vec2 coord;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

void main()
{

    gl_Position = u_proj * u_view * u_model * vec4(vertexPos, 1);
    // gl_Position = u_model * vec4(vertexPos, 1);
    // gl_Position = u_view * u_model * vec4(vertexPos, 1);
    coord = vertexCoord;
}
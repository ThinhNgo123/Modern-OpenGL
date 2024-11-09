#version 460 core

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;

out VS_OUT
{
    vec3 Color;
} vs_out;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

void main()
{
    // gl_Position = u_proj * u_view * u_model * vec4(aPos, 0, 1);
    gl_Position = u_model * vec4(aPos, 0, 1);
    vs_out.Color = aColor;
}
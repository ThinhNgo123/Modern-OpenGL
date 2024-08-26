#version 460 core

layout (location = 0) in vec3 aPos;

out vec3 TexCoord;

uniform mat4 u_view;
uniform mat4 u_proj;

void main()
{
    gl_Position = (u_proj * u_view * vec4(aPos, 1)).xyww;
    // gl_Position = (u_proj * u_view * vec4(aPos, 1));
    TexCoord = aPos;
}
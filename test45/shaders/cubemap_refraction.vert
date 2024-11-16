#version 460 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

// out vec2 TexCoord;
// out VS_OUT
out THINH
{
    vec3 FragPos;
    vec3 Normal;
} vs_out;

layout (std140, binding = 0) uniform Matrices
{
    mat4 u_view;
    mat4 u_proj;
    vec3 eye;
};

uniform mat4 u_model;
// uniform mat4 u_view;
// uniform mat4 u_proj;

// VS_OUT vs_out;

void main()
{

    gl_Position = u_proj * u_view * u_model * vec4(aPos, 1);
    // gl_Position = u_model * vec4(vertexPos, 1);
    // gl_Position = u_view * u_model * vec4(vertexPos, 1);
    // TexCoord = aTexCoord;
    vs_out.FragPos = (u_model * vec4(aPos, 1)).xyz;
    vs_out.Normal = mat3(transpose(inverse(u_model))) * aNormal; 
}
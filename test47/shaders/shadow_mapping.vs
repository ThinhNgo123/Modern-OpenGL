#version 460 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

out VS_OUT 
{
    vec2 TexCoord;
    vec3 Normal;
    vec3 FragPos;
} vs_out;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

void main()
{
    gl_Position = uProj * uView * uModel * vec4(aPos, 1);
    // gl_Position = uModel * vec4(aPos, 1);
    vs_out.TexCoord = aTexCoord;
    vs_out.Normal = transpose(inverse(mat3(uModel))) * aNormal;
    vs_out.FragPos = vec3(uModel * vec4(aPos, 1));
}
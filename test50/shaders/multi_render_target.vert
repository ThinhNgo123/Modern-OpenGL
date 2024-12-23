#version 460 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec2 TexCoords;
out vec3 Normal;
out vec3 FragPos;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform bool uInverseNormal;

void main()
{
    gl_Position = uProj * uView * uModel * vec4(aPos, 1);
    TexCoords = aTexCoords;
    Normal = aNormal * (uInverseNormal ? -1 : 1);
    Normal = transpose(inverse(mat3(uModel))) * Normal;
    FragPos = vec3(uModel * vec4(aPos, 1));
}
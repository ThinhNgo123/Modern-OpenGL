#version 460 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec2 TexCoords;
// out vec3 Normal;
// out vec3 FragPos;

// uniform mat4 uModel;
// uniform mat4 uView;
// uniform mat4 uProj;

void main()
{
    gl_Position = vec4(aPos, 1);
    TexCoords = aTexCoords;
    // Normal = mat3(transpose(inverse(uModel))) * aNormal;
    // FragPos = vec3(uModel * vec4(aPos, 1));
}
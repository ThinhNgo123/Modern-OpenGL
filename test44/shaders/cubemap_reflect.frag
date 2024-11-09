#version 460 core

out vec4 FragColor;

// out vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;

// uniform vec3 eye;

layout (std140, binding = 0) uniform Matrices
{
    mat4 u_view;
    mat4 u_proj;
    vec3 eye;
};

uniform samplerCube texture1;

void main()
{
    vec3 viewDir = normalize(FragPos - eye);
    vec3 normal = normalize(Normal);
    vec3 reflectDir = reflect(viewDir, normal);
    vec4 color = texture(texture1, reflectDir);
    FragColor = color;
}
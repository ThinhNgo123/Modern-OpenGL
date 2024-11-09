#version 460 core

in vec2 TexCoord;

// out vec3 Normal;
out vec4 FragColor;

struct Material
{
    sampler2D texture_diffuse1;
    sampler2D texture_specular01;
};

uniform Material material;

void main()
{
    FragColor = texture(material.texture_diffuse1, TexCoord);
}
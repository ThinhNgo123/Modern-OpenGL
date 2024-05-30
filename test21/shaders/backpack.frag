#version 460 core

in vec2 TextureCoord;

out vec4 fragColor;

// out vec3 FragPos;
// out vec3 Normal;
// out vec2 TextureCoord;

uniform sampler2D diffuse;

void main()
{
    fragColor = texture(diffuse, TextureCoord);
}
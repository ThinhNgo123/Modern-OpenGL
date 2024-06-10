#version 460 core

in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D texture1;
// uniform sampler2D texture2;
// uniform vec3 objectColor;
// uniform vec3 lightColor;

void main()
{
    FragColor = texture(texture1, TexCoord);
}
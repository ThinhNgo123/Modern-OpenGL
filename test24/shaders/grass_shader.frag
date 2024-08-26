#version 460 core

in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D texture1;

void main()
{
    vec4 texture_color = texture(texture1, TexCoord);
    if (texture_color.a < 0.1)
    {
        discard;
    }
    FragColor = texture_color;
}
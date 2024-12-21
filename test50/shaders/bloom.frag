#version 460 core

in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D colorTexture;
uniform sampler2D blurTexture;
uniform float exposure;

void main()
{
    vec3 color = texture(colorTexture, TexCoord).rgb + texture(blurTexture, TexCoord).rgb;
    color = 1 - exp(-color * exposure);
    color = pow(color, vec3(1 / 2.2));
    FragColor = vec4(color, 1);
}
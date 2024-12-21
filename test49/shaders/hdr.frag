#version 460 core

in vec2 TexCoords;
// out vec3 Normal;
// out vec3 FragPos;

out vec4 FragColor;

uniform sampler2D hdrTexture;
uniform float exposure;

void main()
{
    vec3 hdrColor = texture(hdrTexture, TexCoords).rgb;
    // vec3 mappedColor = hdrColor / (hdrColor + vec3(1));
    vec3 mappedColor = 1.0 - exp(-hdrColor * exposure);
    vec3 color = pow(mappedColor, vec3(1 / 2.2));
    FragColor = vec4(color, 1);
}
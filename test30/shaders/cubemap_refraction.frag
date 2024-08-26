#version 460 core

out vec4 FragColor;

// out vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;

uniform vec3 eye;
uniform samplerCube texture1;

vec3 refractCalc(vec3 viewDir, vec3 normal, float ratio)
{
    vec3 ef = ratio * (viewDir + dot(-viewDir, normal) * normal);
    return -sqrt(1 - pow(length(ef), 2)) * normal + ef;
}

void main()
{
    float ratio = 1 / 2.42;
    vec3 viewDir = normalize(FragPos - eye);
    vec3 normal = normalize(Normal);
    vec3 refractDir = refractCalc(viewDir, normal, ratio);
    // vec3 refractDir = refract(viewDir, normal, ratio);
    vec4 color = texture(texture1, refractDir);
    FragColor = color;
}
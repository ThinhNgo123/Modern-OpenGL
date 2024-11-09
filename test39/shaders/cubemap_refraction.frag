#version 460 core

out vec4 FragColor;

// out vec2 TexCoord;
// in VS_OUT
in THINH 
{
    vec3 FragPos;
    vec3 Normal;
} fs_in;

// uniform vec3 eye;

layout (std140, binding = 0) uniform Matrices
{
    mat4 u_view;
    mat4 u_proj;
    vec3 eye;
};

uniform samplerCube texture1;

// VS_OUT fs_in;

vec3 refractCalc(vec3 viewDir, vec3 normal, float ratio)
{
    vec3 ef = ratio * (viewDir + dot(-viewDir, normal) * normal);
    return -sqrt(1 - pow(length(ef), 2)) * normal + ef;
}

void main()
{
    float ratio = 1 / 2.42;
    vec3 viewDir = normalize(fs_in.FragPos - eye);
    vec3 normal = normalize(fs_in.Normal);
    vec3 refractDir = refractCalc(viewDir, normal, ratio);
    // vec3 refractDir = refract(viewDir, normal, ratio);
    vec4 color = texture(texture1, refractDir);
    FragColor = color;
}
#version 460 core

in vec2 TexCoords;
in vec3 Normal;
in vec3 FragPos;

out vec4 FragColor;

#define LIGHT_COUNT 4

struct PointLight
{
    vec3 pos[LIGHT_COUNT];
    vec3 ambient[LIGHT_COUNT];
    vec3 diffuse[LIGHT_COUNT];
    vec3 specular[LIGHT_COUNT];
    float constant;
    float linear;
    float quadratic;
};

uniform sampler2D texture1;
uniform PointLight light;
uniform vec3 viewPos;
uniform float shininess;
uniform bool gammaCorrectionMode;

float gamma = 2.2;

vec3 calcPointLight(vec3 lightPos, vec3 lightAmbient, vec3 lightDiffuse, vec3 lightSpecular)
{
    vec3 diffuseColor = texture(texture1, TexCoords).rgb;
    if (gammaCorrectionMode)
    {
        diffuseColor = pow(diffuseColor, vec3(gamma));
    }

    vec3 lightDir = normalize(lightPos - FragPos);
    vec3 normal = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);

    vec3 ambient = lightAmbient * diffuseColor;
    vec3 diffuse = lightDiffuse * diffuseColor * max(0, dot(lightDir, normal));
    
    vec3 halfVector = normalize(lightDir + viewDir);
    // float eneryConservation = (8 + shininess) / (8 * 3.14);
    float eneryConservation = 1;
    vec3 specular = lightSpecular * eneryConservation * pow(max(0, dot(halfVector, normal)), shininess) * diffuseColor;

    float dis = length(FragPos - lightPos);
    // float attenuation = 1 / (
    //     light.constant + \
    //     (gammaCorrectionMode ? 0 : light.linear) * dis + \
    //     (gammaCorrectionMode ? light.quadratic : 0) * dis * dis
    // );
    float attenuation = 1 / (light.constant + light.linear * dis + light.quadratic * dis * dis);
    
    // return ambient + (diffuse + specular) * attenuation;
    return (diffuse + specular) * attenuation;
}

vec3 gammaCorrection(vec3 color, float gamma)
{
    return pow(color, vec3(1.0 / gamma));
}

void main()
{
    vec3 color = vec3(0);
    for (int i = 0; i < LIGHT_COUNT; i++)
    {
        color += calcPointLight(
            light.pos[i], 
            light.ambient[i],
            light.diffuse[i],
            light.specular[i]
        );
    }

    vec3 diffuseColor = texture(texture1, TexCoords).rgb;
    if (gammaCorrectionMode)
    {
        diffuseColor = pow(diffuseColor, vec3(gamma));
    }

    color += vec3(0.1) * diffuseColor; //ambient
    
    if (gammaCorrectionMode)
    {
        color = gammaCorrection(color, gamma);
    }
    
    FragColor = vec4(color, 1);
}
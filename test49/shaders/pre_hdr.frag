#version 460 core

in vec2 TexCoords;
in vec3 Normal;
in vec3 FragPos;

out vec4 FragColor;

#define LIGHT_COUNT 4

struct PointLight
{
    vec3 pos[LIGHT_COUNT];
    vec3 color[LIGHT_COUNT];
    float constant;
    float linear;
    float quadratic;
};

uniform sampler2D texture1;
uniform vec3 viewPos;
uniform float shininess;
uniform PointLight light;

float gamma = 2.2;

vec3 calcPointLight(vec3 lightPos, vec3 lightColor)
{
    vec3 diffuseColor = texture(texture1, TexCoords).rgb;
    diffuseColor = pow(diffuseColor, vec3(gamma));

    vec3 lightDir = normalize(lightPos - FragPos);
    vec3 normal = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);

    vec3 diffuse = lightColor * diffuseColor * max(0, dot(lightDir, normal));
    
    vec3 halfVector = normalize(lightDir + viewDir);
    // float eneryConservation = (8 + shininess) / (8 * 3.14);
    vec3 specularColor = vec3(0);
    vec3 specular = lightColor * specularColor * pow(max(0, dot(halfVector, normal)), shininess);

    float dis = length(FragPos - lightPos);
    // float attenuation = 1 / (
    //     light.constant + 
    //     (gammaCorrectionMode ? 0 : light.linear) * dis + 
    //     (gammaCorrectionMode ? light.quadratic : 0) * dis * dis
    // );
    float attenuation = 1 / (dis * dis);
    
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
        color += calcPointLight(light.pos[i], light.color[i]);
    }
    
    // color = gammaCorrection(color, gamma);
    
    FragColor = vec4(color, 1);
    // FragColor = vec4(1, 0, 0, 1);
}
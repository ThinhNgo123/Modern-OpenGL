#version 460 core

in vec2 TexCoords;
in vec3 Normal;
in vec3 FragPos;

out vec4 FragColor;

struct PointLight
{
    vec3 pos;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float constant;
    float linear;
    float quadratic;
};

uniform sampler2D texture1;
uniform PointLight light;
uniform vec3 viewPos;
uniform float shininess;
uniform bool lightMode;

vec3 calcPointLight()
{
    vec3 diffuseColor = texture(texture1, TexCoords).rgb;

    vec3 lightDir = normalize(light.pos - FragPos);
    vec3 normal = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);

    vec3 ambient = light.ambient * diffuseColor;
    vec3 diffuse = light.diffuse * diffuseColor * max(0, dot(lightDir, normal));
    
    vec3 specular = vec3(0);
    vec3 halfVector = vec3(0);
    if (lightMode)
    {
        const float energyConservation = (2 + shininess) / (2 * 3.14);
        specular += light.specular * energyConservation * pow(max(0, dot(reflect(-lightDir, normal), viewDir)), shininess);
    }
    else
    {
        const float energyConservation = (8 + shininess) / (8 * 3.14);
        halfVector += normalize(lightDir + viewDir);
        specular += light.specular * energyConservation * pow(max(0, dot(halfVector, normal)), shininess + 32);
    }

    float dis = distance(light.pos, FragPos);
    float attenution = 1 / (light.constant + light.linear * dis + light.quadratic * dis * dis);

    return ambient + (diffuse + specular) * attenution;
}

void main()
{
    FragColor = vec4(calcPointLight(), 1);
}
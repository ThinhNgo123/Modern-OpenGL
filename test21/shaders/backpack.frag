#version 460 core

in vec3 FragPos;
in vec3 Normal;
in vec2 TextureCoord;

out vec4 FragColor;

struct Material
{
    sampler2D texture_diffuse1;
    sampler2D texture_specular1;
    float shininess;
};

struct PointLight
{
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float constant;
    float linear;
    float quadratic;
};

uniform Material material;
uniform PointLight pointLight;
uniform vec3 viewPos;

void main()
{
    vec3 Normal = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 lightDir = normalize(pointLight.position - FragPos);

    vec3 ambient = pointLight.ambient * texture(material.texture_diffuse1, TextureCoord).rgb;

    vec3 diffuse = pointLight.diffuse * texture(material.texture_diffuse1, TextureCoord).rgb * max(dot(lightDir, Normal), 0);

    vec3 reflectDir = reflect(-lightDir, Normal);
    vec3 specular = pointLight.specular * texture(material.texture_specular1, TextureCoord).rgb * pow(max(dot(reflectDir, viewDir), 0), material.shininess);

    float dis = length(pointLight.position - FragPos);
    float attenuation = 1 / (pointLight.constant + pointLight.linear * dis + pointLight.quadratic * dis * dis);

    FragColor = vec4((ambient + diffuse + specular) * attenuation, 1);
}
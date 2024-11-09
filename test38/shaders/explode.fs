#version 460 core

in VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    vec2 TextureCoord;
} fs_in;

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

struct DirectionLight
{
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

uniform Material material;
// uniform PointLight pointLight;
uniform DirectionLight directionLight;
uniform vec3 viewPos;

void main()
{
    vec3 Normal = normalize(fs_in.Normal);
    vec3 viewDir = normalize(viewPos - fs_in.FragPos);
    // vec3 lightDir = normalize(pointLight.position - fs_in.FragPos);
    vec3 lightDir = normalize(directionLight.direction);

    vec3 ambient = directionLight.ambient * texture(material.texture_diffuse1, fs_in.TextureCoord).rgb;

    vec3 diffuse = directionLight.diffuse * texture(material.texture_diffuse1, fs_in.TextureCoord).rgb * max(dot(lightDir, Normal), 0);

    vec3 reflectDir = reflect(-lightDir, Normal);
    vec3 specular = directionLight.specular * texture(material.texture_specular1, fs_in.TextureCoord).rgb * pow(max(dot(reflectDir, viewDir), 0), material.shininess);

    // float dis = length(pointLight.position - fs_in.FragPos);
    // float attenuation = 1 / (pointLight.constant + pointLight.linear * dis + pointLight.quadratic * dis * dis);

    // FragColor = vec4((ambient + diffuse + specular) * attenuation, 1);
    FragColor = vec4(ambient + diffuse + specular, 1);
}
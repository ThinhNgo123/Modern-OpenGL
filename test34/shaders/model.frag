#version 460 core

in vec3 Normal;
in vec3 FragPos;

out vec4 FragColor;

struct Material
{
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

struct DirLight
{
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

uniform vec3 viewPos;
uniform DirLight dirLight;
uniform Material material;

void main()
{
    float shininess = 10;
    // vec3 colorObject = vec3(float(215) / 255);
    vec3 Normal = normalize(Normal);
    vec3 direction = normalize(dirLight.direction);
    vec3 viewDir = normalize(viewPos - FragPos); 

    vec3 ambient = dirLight.ambient * material.ambient;
    vec3 diffuse = dirLight.diffuse * material.diffuse * max(dot(-direction, Normal), 0);
    vec3 reflectVec = reflect(direction, Normal);
    vec3 specular = dirLight.specular * material.specular * pow(max(dot(reflectVec, viewDir), 0), material.shininess);
    FragColor = vec4(ambient + diffuse + specular, 1);
}
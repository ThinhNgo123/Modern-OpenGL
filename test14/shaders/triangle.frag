#version 460 core

in vec3 FragPos;
in vec3 Normal;

out vec4 fragColor;

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

struct Light {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

// uniform sampler2D texture1;
// uniform sampler2D texture2;
uniform vec3 viewPos;
uniform Material material;
uniform Light light;

void main()
{
    vec3 ambient = light.ambient * material.ambient;

    vec3 lightDir = normalize(light.position - FragPos);
    vec3 diffuse = light.diffuse * material.diffuse * max(dot(lightDir, Normal), 0);

    vec3 reflectDir = reflect(-lightDir, Normal);
    vec3 specular = light.specular * material.specular * pow(max(dot(reflectDir, normalize(viewPos - FragPos)), 0), material.shininess);
    
    fragColor = vec4(ambient + diffuse + specular, 1);
}
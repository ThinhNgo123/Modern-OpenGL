#version 460 core

in vec3 FragPos;
in vec3 Normal;
in vec2 TextureCoord;

out vec4 fragColor;

struct Material {
    sampler2D diffuse;
    sampler2D specular;
    sampler2D emission;
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
    vec3 ambient = light.ambient * vec3(texture(material.diffuse, TextureCoord));

    vec3 lightDir = normalize(light.position - FragPos);
    vec3 diffuse = light.diffuse * vec3(texture(material.diffuse, TextureCoord)) * max(dot(lightDir, Normal), 0);

    vec3 reflectDir = reflect(-lightDir, Normal);
    vec3 specular = light.specular * vec3(texture(material.specular, TextureCoord)) * pow(max(dot(reflectDir, normalize(viewPos - FragPos)), 0), material.shininess);
    
    vec3 emission = vec3(0);
    vec3 specular_color = vec3(texture(material.specular, TextureCoord));
    if (specular_color.x < 0.02 && specular_color.y < 0.02 && specular_color.z < 0.02)
    {
        emission += vec3(texture(material.emission, TextureCoord)) * max(dot(lightDir, Normal), vec3(texture(material.diffuse, TextureCoord)).x);
    }

    fragColor = vec4(ambient + diffuse + specular + emission, 1);
}
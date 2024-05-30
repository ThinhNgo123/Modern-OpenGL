#version 460 core

in vec3 FragPos;
in vec3 Normal;

out vec4 fragColor;

// uniform sampler2D texture1;
// uniform sampler2D texture2;
uniform vec3 objectColor;
uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 viewPos;

void main()
{
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;
    vec3 lightDir = normalize(lightPos - FragPos);
    vec3 diffuse = max(dot(lightDir, Normal), 0) * lightColor;
    vec3 reflectDir = reflect(-lightDir, Normal);
    float specularStrength = 0.9;
    vec3 specular = specularStrength * pow(max(dot(reflectDir, normalize(viewPos - FragPos)), 0), 32) * lightColor;
    fragColor = vec4((ambient + diffuse + specular) * objectColor, 1);
}
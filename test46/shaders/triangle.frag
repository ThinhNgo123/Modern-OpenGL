#version 460 core

#define NUMBER_POINT_LIGHT 4

// in vec3 FragPos;
// in vec3 Normal;
in vec2 TextureCoord;

out vec4 fragColor;

struct Material {
    sampler2D texture_diffuse2;
    sampler2D texture_specular2;
    // sampler2D emission;
    float shininess;
};

struct DirLight
{
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
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

struct SpotLight
{
    vec3 position;
    vec3 direction;
    float cutOff;      // cos(phi)
    float outerCutOff; // cos(gamma)
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float constant;
    float linear;
    float quadratic;
};

// uniform sampler2D texture1;
// uniform sampler2D texture2;
// uniform vec3 viewPos;
uniform Material material;
// uniform DirLight dirLight;
// uniform PointLight pointLight[NUMBER_POINT_LIGHT];
// uniform SpotLight spotLight;

// vec3 calcDirectionLight(DirLight light, vec3 normal, vec3 viewDir)
// {
//     vec3 lightDir = normalize(-light.direction);
//     normal = normalize(normal);
//     viewDir = normalize(viewDir);
//     vec3 ambient = light.ambient * texture(material.diffuse, TextureCoord).rgb;
//     vec3 diffuse = light.diffuse * texture(material.diffuse, TextureCoord).rgb * max(dot(lightDir, normal), 0);
//     vec3 reflectDir = reflect(-lightDir, normal);
//     vec3 specular = light.specular * texture(material.specular, TextureCoord).rgb * pow(max(dot(reflectDir, viewDir), 0), material.shininess);
//     return ambient + diffuse + specular;
// }

// vec3 calcPointLight(PointLight light, vec3 normal, vec3 viewDir)
// {
//     vec3 lightDir = normalize(light.position - FragPos);
//     normal = normalize(normal);
//     viewDir = normalize(viewDir);
//     vec3 ambient = light.ambient * texture(material.diffuse, TextureCoord).rgb;
//     vec3 diffuse = light.diffuse * texture(material.diffuse, TextureCoord).rgb * max(dot(lightDir, normal), 0);
//     vec3 reflectDir = reflect(-lightDir, normal);
//     vec3 specular = light.specular * texture(material.specular, TextureCoord).rgb * pow(max(dot(reflectDir, viewDir), 0), material.shininess);

//     float dis = distance(light.position, FragPos);
//     float attenution = 1 / (light.constant + light.linear * dis + light.quadratic * dis * dis);

//     return (ambient + diffuse + specular) * attenution;
// }

// vec3 calcSpotLight(SpotLight light, vec3 normal, vec3 viewDir)
// {
//     vec3 lightDir = normalize(light.position - FragPos);
//     normal = normalize(normal);
//     viewDir = normalize(viewDir);
//     vec3 ambient = light.ambient * texture(material.diffuse, TextureCoord).rgb;
//     vec3 diffuse = light.diffuse * texture(material.diffuse, TextureCoord).rgb * max(dot(lightDir, normal), 0);
//     vec3 reflectDir = reflect(-lightDir, normal);
//     vec3 specular = light.specular * texture(material.specular, TextureCoord).rgb * pow(max(dot(reflectDir, viewDir), 0), material.shininess);

//     float dis = distance(light.position, FragPos);
//     float attenution = 1 / (light.constant + light.linear * dis + light.quadratic * dis * dis);

//     float cos_theta = dot(normalize(light.direction), -lightDir);
//     float intensity = clamp((cos_theta - light.outerCutOff) / (light.cutOff - light.outerCutOff), 0, 1);

//     return (ambient + diffuse + specular) * attenution * intensity;
// }

void main()
{
    // vec3 result = vec3(0);
    // result += calcDirectionLight(dirLight, Normal, viewPos - FragPos);
    
    // for (int i = 0; i < NUMBER_POINT_LIGHT; i++)
    // {
    //     result += calcPointLight(pointLight[i], Normal, viewPos - FragPos);
    // }

    // result += calcSpotLight(spotLight, Normal, viewPos - FragPos);
    
    // vec3 emission = vec3(0);
    // vec3 specular_color = vec3(texture(material.specular, TextureCoord));
    // if (specular_color.x < 0.02 && specular_color.y < 0.02 && specular_color.z < 0.02)
    // {
    //     emission += vec3(texture(material.emission, TextureCoord)) * max(dot(lightDir, Normal), vec3(texture(material.diffuse, TextureCoord)).x);
    // }

    // fragColor = vec4(ambient + diffuse + specular + emission, 1);
    
    fragColor = vec4(texture(material.texture_diffuse2, TextureCoord).rgb, 1);
} 
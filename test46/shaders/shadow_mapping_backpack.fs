#version 460 core

in VS_OUT
{
    vec2 TexCoord;
    vec3 Normal;
    vec3 FragPos;
} fs_in;

out vec4 FragColor;

struct Material
{
    sampler2D texture_diffuse1;
    sampler2D texture_specular1;
};

uniform sampler2D depthMap;
// uniform sampler2D diffuseMap;
uniform Material material;
uniform mat4 uLightSpace;
uniform vec3 uLightColor;
uniform vec3 uLightDir;
uniform vec3 uViewPos;

void main()
{
    //calc shadow
    vec4 fragPosLightSpace = uLightSpace * vec4(fs_in.FragPos, 1);
    vec3 fragCoord = fragPosLightSpace.xyz / fragPosLightSpace.w;
    fragCoord = 0.5 * fragCoord + 0.5; 

    // PCF
    // float closestDepth = texture(depthMap, fragCoord.xy).r;
    vec2 texelDepthSize = 1.0 / textureSize(depthMap, 0);

    // float bias = 0.005;
    // float bias = (0.05 * (1 - dot(normalize(fs_in.Normal), normalize(uLightPos - fragPosLightSpace.xyz))), 0.005);
    float bias = max(0.05 * (1 - dot(normalize(fs_in.Normal), normalize(-uLightDir))), 0.005);
    // float bias = 0.003;

    float shadowValue = 0;
    int count = 2;
    for (int i = -count; i <= count; i++)
    {
        for (int j = -count; j <= count; j++)
        {
            float pcfDepth = texture(depthMap, fragCoord.xy + vec2(i, j) * texelDepthSize).r;
            shadowValue += (fragCoord.z - bias > pcfDepth) ? 1 : 0;
        }
    }
    shadowValue /= (2 * count + 1) * (2 * count + 1);

    if (fragCoord.z > 1.0)
    {
        shadowValue = 0.0;
    }
    
    // calc lighting
    // vec3 color = texture(diffuseMap, fs_in.TexCoord).rgb;
    vec3 diffuseColor = texture(material.texture_diffuse1, fs_in.TexCoord).rgb;
    vec3 specularColor = texture(material.texture_specular1, fs_in.TexCoord).rgb;

    vec3 lightDir = normalize(-uLightDir);
    vec3 normal = normalize(fs_in.Normal);
    vec3 viewDir = normalize(uViewPos - fs_in.FragPos);

    vec3 ambient, diffuse, specular;

    ambient = 0.3 * uLightColor;

    diffuse = uLightColor * max(0, dot(lightDir, normal));

    vec3 halfwayDir = normalize(lightDir + viewDir);
    specular = uLightColor * pow(max(0, dot(halfwayDir, normal)), 64);

    FragColor = vec4((ambient * diffuseColor + (1 - shadowValue) * (diffuse * diffuseColor + specular * specularColor)), 1);
    // FragColor = vec4(color * (ambient + (1 - shadowValue)), 1);
}
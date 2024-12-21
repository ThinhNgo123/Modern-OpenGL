#version 460 core

in VS_OUT
{
    vec2 TexCoord;
    vec3 Normal;
    vec3 FragPos;
} fs_in;

out vec4 FragColor;

uniform sampler2D depthMap;
uniform sampler2D diffuseMap;
uniform mat4 uLightSpace;
uniform vec3 uLightColor;
uniform vec3 uLightDir;
uniform vec3 uViewPos;

float sampleShadowLinear(sampler2D sampleTexure, vec3 texCoords, vec2 texelSize, float bias)
{
    ivec2 intergerPart = ivec2(texCoords.xy / texelSize);
    vec2 fractPart = fract(texCoords.xy - intergerPart);
    float bottomLeft = step(texture(sampleTexure, (intergerPart + vec2(0.5, 0.5)) * texelSize).r, texCoords.z - bias);
    float bottomRight = step(texture(sampleTexure, (intergerPart + vec2(1.5, 0.5)) * texelSize).r, texCoords.z - bias);
    float topLeft = step(texture(sampleTexure, (intergerPart + vec2(0.5, 1.5)) * texelSize).r, texCoords.z - bias);
    float topRight = step(texture(sampleTexure, (intergerPart + vec2(1.5, 1.5)) * texelSize).r, texCoords.z - bias);
    float mixTopBottomLeft = mix(bottomLeft, topLeft, fractPart.y);
    float mixTopBottomRight = mix(bottomRight, topRight, fractPart.y);
    return mix(mixTopBottomLeft, mixTopBottomRight, fractPart.x);
}

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
    // float bias = texelDepthSize.x;

    float shadowValue = 0;
    int count = 1;
    for (int i = -count; i <= count; i++)
    {
        for (int j = -count; j <= count; j++)
        {
            float pcfDepth = texture(depthMap, fragCoord.xy + vec2(i, j) * texelDepthSize).r;
            shadowValue += (fragCoord.z - bias > pcfDepth) ? 1 : 0;
            // shadowValue += sampleShadowLinear(depthMap, fragCoord + vec3(i * texelDepthSize.x , j * texelDepthSize.y , 0), texelDepthSize, bias);
        }
    }
    shadowValue /= (2 * count + 1) * (2 * count + 1);

    if (fragCoord.z > 1.0)
    {
        shadowValue = 0.0;
    }
    
    // calc lighting
    vec3 color = texture(diffuseMap, fs_in.TexCoord).rgb;

    vec3 lightDir = normalize(-uLightDir);
    vec3 normal = normalize(fs_in.Normal);
    vec3 viewDir = normalize(uViewPos - fs_in.FragPos);

    vec3 ambient, diffuse, specular;

    ambient = 0.3 * uLightColor;

    diffuse = uLightColor * max(0, dot(lightDir, normal));

    vec3 halfwayDir = normalize(lightDir + viewDir);
    specular = uLightColor * pow(max(0, dot(halfwayDir, normal)), 64);

    FragColor = vec4(color * (ambient + (1 - shadowValue) * (diffuse + specular)), 1);
    // FragColor = vec4(color * (ambient + (1 - shadowValue)), 1);
}
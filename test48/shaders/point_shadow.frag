#version 460 core

in vec3 Normal;
in vec2 TexCoord;
in vec3 FragPos;

out vec4 FragColor;

uniform vec3 pointLightPos;
uniform vec3 pointLightColor;
uniform float ambientLight;
uniform float diffuseLight;
uniform float specularLight;
uniform samplerCube textureCube;
uniform sampler2D texture1;
uniform vec3 viewPos;
uniform float farLight;

float calcPointShadow(samplerCube textureCube, vec3 lightDir, float farLight)
{
    float depthCurrent = length(lightDir) / farLight;
    
    float shadow = 0;
    float samples = 4;
    float offset = 0.1;
    for(float x = -offset; x < offset; x += offset / (samples * 0.5)) 
    {
        for(float y = -offset; y < offset; y += offset / (samples * 0.5)) 
        {
            for(float z = -offset; z < offset; z += offset / (samples * 0.5)) 
            {
                float depthCloset = texture(textureCube, lightDir + vec3(x, y, z)).r;
                shadow += step(depthCloset, depthCurrent - 0.05);
            }
        }
    }
    
    return shadow / (samples * samples * samples);
}

void main()
{
    vec3 colorTexture = texture(texture1, TexCoord).rgb;
    colorTexture = pow(colorTexture, vec3(2.2));

    // vec3 normal = normalize(Normal);
    vec3 normal = normalize((int(gl_FrontFacing) - 0.5) * 2 * Normal);
    // if (!gl_FrontFacing)
    // {
    //     normal = -normal;
    // }
    vec3 lightDir = FragPos - pointLightPos;
    vec3 lightDirUnit = normalize(lightDir);
    vec3 viewDir = normalize(viewPos - FragPos);

    vec3 ambient = pointLightColor * ambientLight;

    vec3 diffuse = pointLightColor * diffuseLight * max(0, dot(-lightDirUnit, normal));

    vec3 halfDir = normalize(-lightDirUnit + viewDir);
    vec3 specular = pointLightColor * specularLight * pow(max(0, dot(normal, halfDir)), 32);

    float shadowFactor = calcPointShadow(textureCube, lightDir, farLight);
    vec3 color = (ambient + (1 - shadowFactor) * (diffuse + specular)) * colorTexture;
    // vec3 color = vec3(1, 0.5, 0.3);

    color = pow(color, vec3(1 / 2.2));

    FragColor = vec4(color, 1);
}
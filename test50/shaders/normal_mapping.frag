#version 460 core

in vec3 Normal;
in vec2 TexCoord;
in vec3 FragPos;

out vec4 FragColor;

uniform vec3 pointLightPos;
// uniform vec3 pointLightColor;
uniform vec3 ambientLight;
uniform vec3 diffuseLight;
uniform vec3 specularLight;
// uniform samplerCube textureCube;
uniform sampler2D diffuseTexture;
uniform sampler2D normalTexture;
uniform vec3 viewPos;
// uniform float farLight;

void main()
{
    vec3 colorTexture = texture(diffuseTexture, TexCoord).rgb;
    // colorTexture = pow(colorTexture, vec3(2.2));

    // vec3 normal = normalize(Normal);
    // vec3 normal = normalize((int(gl_FrontFacing) - 0.5) * 2 * Normal);
    vec3 normal = normalize(texture(normalTexture, TexCoord).rgb * 2 - 1);
    // if (!gl_FrontFacing)
    // {
    //     normal = -normal;
    // }
    vec3 lightDir = FragPos - pointLightPos;
    vec3 lightDirUnit = normalize(lightDir);
    vec3 viewDir = normalize(viewPos - FragPos);

    vec3 ambient = ambientLight;

    vec3 diffuse = diffuseLight * max(0, dot(-lightDirUnit, normal));

    vec3 halfDir = normalize(-lightDirUnit + viewDir);
    vec3 specular = specularLight * pow(max(0, dot(normal, halfDir)), 32);

    // vec3 color = (ambient + (1 - shadowFactor) * (diffuse + specular)) * colorTexture;
    vec3 color = (ambient + diffuse) * colorTexture + specular;
    // vec3 color = vec3(1, 0.5, 0.3);

    // color = pow(color, vec3(1 / 2.2));

    FragColor = vec4(color, 1);
}
#version 460 core

in vec2 TexCoord;

out vec4 FragColor;

uniform bool isVertical;
uniform sampler2D texture1;
uniform vec2 texelSize;

float gaussWeight[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main()
{
    vec3 color = texture(texture1, TexCoord).rgb * gaussWeight[0];
    if (isVertical)
    {
        for (int i = 1; i < 5; i++)
        {
            color += texture(texture1, TexCoord + vec2(0, texelSize.y * i)).rgb * gaussWeight[i];
            color += texture(texture1, TexCoord + vec2(0, -texelSize.y * i)).rgb * gaussWeight[i];
        }
    }
    else
    {
        for (int i = 1; i < 5; i++)
        {
            color += texture(texture1, TexCoord + vec2(texelSize.x * i, 0)).rgb * gaussWeight[i];
            color += texture(texture1, TexCoord + vec2(-texelSize.x * i, 0)).rgb * gaussWeight[i];
        }
    }
    FragColor = vec4(color, 1);
}
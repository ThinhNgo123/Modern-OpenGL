#version 460 core

// in vec2 coord;

out vec4 FragColor;
out vec4 BrightColor;

// uniform sampler2D texture1;
// uniform sampler2D texture2;
// uniform vec3 objectColor;
uniform vec3 lightColor;

void main()
{
    // fragColor = vec4(1, 1, 1, 1);
    FragColor = vec4(lightColor, 1);

    float brightness = dot(lightColor, vec3(0.2126, 0.7152, 0.0722));
    BrightColor = vec4(lightColor * step(1.0, brightness), 1);
}
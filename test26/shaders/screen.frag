#version 460 core

in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D screenTexture;
uniform float offset;

vec2 offsets[9] = vec2[]
(
    vec2(-offset,  offset),
    vec2(      0,  offset),
    vec2( offset,  offset),
    vec2(-offset,       0),
    vec2(      0,       0),
    vec2( offset,       0),
    vec2(-offset, -offset),
    vec2(      0, -offset),
    vec2( offset, -offset)
);

float kernel[9] = float[]
(
    2,   2, 2,
    2, -15, 2,
    2,   2, 2
);

float blurKernel[9] = float[]
(
    1 / 16.0, 2 / 16.0, 1 / 16.0,
    2 / 16.0, 4 / 16.0, 2 / 16.0,
    1 / 16.0, 2 / 16.0, 1 / 16.0
);

float edgeDetectionKernel[9] = float[]
(
    1,  1, 1,
    1, -8, 1,
    1,  1, 1
);

void main()
{
    // normal
    // FragColor = texture(screenTexture, TexCoord);

    // inverse
    // FragColor = vec4(1 - texture(screenTexture, TexCoord).rgb, 1);

    // grayscale
    // vec3 texture = texture(screenTexture, TexCoord).rgb;
    // FragColor = vec4(vec3((0.2126 * texture.r + 0.7152 * texture.g + 0.0722 * texture.b)), 1);

    // blur
    // vec3 color = vec3(0.0);
    // for(int i = 0; i < 9; i++)
    // {
    //     color += blurKernel[i] * texture(screenTexture, TexCoord + offsets[i]).rgb;
    // }
    // FragColor = vec4(color, 1);

    // edge detection
    vec3 color = vec3(0.0);
    for(int i = 0; i < 9; i++)
    {
        color += edgeDetectionKernel[i] * texture(screenTexture, TexCoord + offsets[i]).rgb;
    }
    FragColor = vec4(color, 1);
}
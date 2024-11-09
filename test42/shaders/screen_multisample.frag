#version 460 core

in vec3 TexCoord;

out vec4 FragColor;

uniform sampler2DMS screenTextureMS;
// uniform float offset;

// vec2 offsets[9] = vec2[]
// (
//     vec2(-offset,  offset),
//     vec2(      0,  offset),
//     vec2( offset,  offset),
//     vec2(-offset,       0),
//     vec2(      0,       0),
//     vec2( offset,       0),
//     vec2(-offset, -offset),
//     vec2(      0, -offset),
//     vec2( offset, -offset)
// );

// float kernel[9] = float[]
// (
//     2,   2, 2,
//     2, -15, 2,
//     2,   2, 2
// );

// float blurKernel[9] = float[]
// (
//     1 / 16.0, 2 / 16.0, 1 / 16.0,
//     2 / 16.0, 4 / 16.0, 2 / 16.0,
//     1 / 16.0, 2 / 16.0, 1 / 16.0
// );

// float edgeDetectionKernel[9] = float[]
// (
//     1,  1, 1,
//     1, -8, 1,
//     1,  1, 1
// );

void main()
{
    // normal
    // FragColor = texture(screenTexture, TexCoord.xy);
    ivec2 size = textureSize(screenTextureMS);
    // ivec2 size = ivec2(width, height);
    vec4 colorSample0 = texelFetch(screenTextureMS, ivec2(TexCoord.xy * size), 0);
    vec4 colorSample1 = texelFetch(screenTextureMS, ivec2(TexCoord.xy * size), 1);
    vec4 colorSample2 = texelFetch(screenTextureMS, ivec2(TexCoord.xy * size), 2);
    vec4 colorSample3 = texelFetch(screenTextureMS, ivec2(TexCoord.xy * size), 3);
    // if (size.x == 1200 && size.y == 800)
    // {
    //     FragColor = vec4(1, 0, 0, 1);
    // }
    // else
    // {
    //     FragColor = vec4(0, 1, 0, 1);
    // } 
    FragColor = (colorSample0 + colorSample1 + colorSample2 + colorSample3) / 4;
    // FragColor = colorSample0;
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
    // vec3 color = vec3(0.0);
    // for(int i = 0; i < 9; i++)
    // {
    //     color += edgeDetectionKernel[i] * texture(screenTexture, TexCoord + offsets[i]).rgb;
    // }
    // FragColor = vec4(color, 1);
}
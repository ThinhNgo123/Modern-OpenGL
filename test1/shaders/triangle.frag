#version 460 core

in vec4 pos;
in vec2 texCoord;

out vec4 fragColor;

uniform sampler2D texture0;
uniform sampler2D texture1;
uniform float mixScale;

void main()
{
    // fragColor = vec4(pos.x, 0.5, 0.5, 1);
    // fragColor = mix(
    //     texture(texture0, texCoord),
    //     texture(texture1, texCoord),
    //     mixScale
    // );
    fragColor = texture(texture0, texCoord) + \
        texture(texture1, texCoord) * mixScale;
}
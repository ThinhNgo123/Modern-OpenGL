#version 460 core

in vec2 coord;

out vec4 fragColor;

uniform sampler2D texture1;
uniform sampler2D texture2;

void main()
{
    // fragColor = vec4(pos.x, 0.5, 0.5, 1);
    fragColor = mix(
        texture(texture1, coord),
        texture(texture2, coord),
        0.2
    );
    // fragColor = texture(texture1, coord) + \
        // texture(texture2, coord) * mixScale;
    // fragColor = vec4(color, 1);
}
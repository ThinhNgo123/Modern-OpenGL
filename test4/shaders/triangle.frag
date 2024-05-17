#version 460 core

in vec4 pos;
in vec3 color;

out vec4 fragColor;

void main()
{
    // fragColor = vec4(pos.x, 0.5, 0.5, 1);
    // fragColor = mix(
    //     texture(texture0, texCoord),
    //     texture(texture1, texCoord),
    //     mixScale
    // );
    // fragColor = texture(texture0, texCoord) + \
        // texture(texture1, texCoord) * mixScale;
    fragColor = vec4(color, 1);
}
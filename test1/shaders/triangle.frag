#version 460 core

in vec4 pos;

out vec4 fragColor;

void main()
{
    fragColor = vec4(pos.x, 0.5, 0.5, 1);
    // fragColor = vec4(pos.x, pos.x, 0.5, 1);
}
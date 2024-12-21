#version 460 core

out vec4 FragColor;

void main()
{
    // vec4 texture_color = texture(texture1, TexCoord);
    // if (texture_color.a < 0.1)
    // {
    //     discard;
    // }
    // FragColor = texture_color;
    FragColor = vec4(0, 1, 0, 1);
}
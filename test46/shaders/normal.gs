#version 460 core

layout (triangles) in;
layout (line_strip, max_vertices = 2) out;

in VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    vec2 TextureCoord;
} gs_in[];

void draw_normal(int index)
{
    gl_Position = gl_in[index].gl_Position;
    EmitVertex();
    gl_Position = gl_in[index].gl_Position + vec4(-normalize(gs_in[index].Normal), 0) * 0.4;
    EmitVertex();
    EndPrimitive();
}

void main()
{
    for (int i = 0; i < 3; i++)
    {
        draw_normal(i);
    }
}
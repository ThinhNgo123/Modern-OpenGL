#version 460 core

layout (points) in;
layout (triangle_strip, max_vertices = 6) out;

in VS_OUT
{
    vec3 Color;
} gs_in[];

out vec3 Color;

void geometry_house(vec4 pos)
{
    Color = gs_in[0].Color;
    gl_Position = pos + vec4(-0.2, -0.2, 0, 0);
    EmitVertex();
    gl_Position = pos + vec4(0.2, -0.2, 0, 0);
    EmitVertex();
    gl_Position = pos + vec4(-0.2, 0.2, 0, 0);
    EmitVertex();
    gl_Position = pos + vec4(0.2, 0.2, 0, 0);
    EmitVertex();
    Color = vec3(1, 1, 1);
    gl_Position = pos + vec4(0, 0.4, 0, 0);
    EmitVertex();
    EndPrimitive();
}

void main()
{
    geometry_house(gl_in[0].gl_Position);
    // gl_Position = gl_in[0].gl_Position;
    // EmitVertex();
    // EndPrimitive();
}
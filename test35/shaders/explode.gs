#version 460 core

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    vec2 TextureCoord;
} gs_in[];

out VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    vec2 TextureCoord;
} gs_out;

uniform float time;

vec3 getNormalFromTriangle(vec3 pos1, vec3 pos2, vec3 pos3)
{
    return normalize(cross(pos3 - pos1, pos2 - pos1));
}

void explode(vec4 pos1, vec4 pos2, vec4 pos3, float time)
{
    vec3 direction = getNormalFromTriangle(pos1.xyz, pos2.xyz, pos3.xyz);
    vec3 velocity = direction * ((sin(time) + 1) / 2) * 2;
    gs_out.FragPos = gs_in[0].FragPos;
    gs_out.Normal = gs_in[0].Normal;
    gs_out.TextureCoord = gs_in[0].TextureCoord;
    gl_Position = pos1 + vec4(velocity, 1);
    EmitVertex();
    gs_out.FragPos = gs_in[1].FragPos;
    gs_out.Normal = gs_in[1].Normal;
    gs_out.TextureCoord = gs_in[1].TextureCoord;
    gl_Position = pos2 + vec4(velocity, 1);
    EmitVertex();
    gs_out.FragPos = gs_in[2].FragPos;
    gs_out.Normal = gs_in[2].Normal;
    gs_out.TextureCoord = gs_in[2].TextureCoord;
    gl_Position = pos3 + vec4(velocity, 1);
    EmitVertex();
    EndPrimitive();
}

// void geometry_house(vec4 pos)
// {
//     Color = gs_in[0].Color;
//     gl_Position = pos + vec4(-0.2, -0.2, 0, 0);
//     EmitVertex();
//     gl_Position = pos + vec4(0.2, -0.2, 0, 0);
//     EmitVertex();
//     gl_Position = pos + vec4(-0.2, 0.2, 0, 0);
//     EmitVertex();
//     gl_Position = pos + vec4(0.2, 0.2, 0, 0);
//     EmitVertex();
//     Color = vec3(1, 1, 1);
//     gl_Position = pos + vec4(0, 0.4, 0, 0);
//     EmitVertex();
//     EndPrimitive();
// }

void main()
{
    // geometry_house(gl_in[0].gl_Position);
    // gl_Position = gl_in[0].gl_Position;
    // EmitVertex();
    // EndPrimitive();
    explode(
        gl_in[0].gl_Position, 
        gl_in[1].gl_Position, 
        gl_in[2].gl_Position,
        time
    );
}


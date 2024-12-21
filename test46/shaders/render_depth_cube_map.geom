#version 460 core

layout (triangles) in;
layout (triangle_strip, max_vertices = 18) out;

out vec3 FragPos;

uniform mat4 uViewLight[6];
uniform mat4 uProjLight;

void main()
{
    //right, left, top, bottom, back, front
    for (int face = 0; face < 6; face++)
    {
        gl_Layer = face;
        for (int i = 0; i < 3; i++)
        {
            gl_Position = uProjLight * uViewLight[face] * gl_in[i].gl_Position;
            FragPos = vec3(gl_in[i].gl_Position);
            EmitVertex();
        }
        EndPrimitive();
    }
}
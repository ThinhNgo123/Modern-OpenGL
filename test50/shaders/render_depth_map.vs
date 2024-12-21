#version 460 core

layout(location = 0) in vec3 aPos;

uniform mat4 uModel;
uniform mat4 uLightSpace;
// uniform mat4 uView;
// uniform mat4 uProj;
// uniform mat4 uVP;

void main()
{
    gl_Position = uLightSpace * uModel * vec4(aPos, 1);
    // gl_Position = uVP * uModel * vec4(aPos, 1);
}
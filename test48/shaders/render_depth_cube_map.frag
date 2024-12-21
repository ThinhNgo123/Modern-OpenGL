#version 460 core

in vec3 FragPos;

// uniform mat4 uModel;
// uniform mat4 uView;
// uniform mat4 uProj;
// uniform sampler2D uTexture;
uniform vec3 pointLightPos;
uniform float farLight;

void main()
{
    gl_FragDepth = distance(FragPos, pointLightPos) / farLight;
}
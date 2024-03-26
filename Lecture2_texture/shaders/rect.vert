#version 120 core

attribute vec2 position;
attribute vec3 color;
attribute vec2 inTexCoord;

varying vec3 outColor;
varying vec2 outTexCoord;

void main()
{
    gl_Position = vec4(position, 0, 1);
    outColor = color;
    outTexCoord = inTexCoord;
}
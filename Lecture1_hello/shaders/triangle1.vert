#version 120 core

attribute vec4 vertexPos;

//varying vec3 fragmentColor;

void main()
{
    gl_Position = vertexPos;
}
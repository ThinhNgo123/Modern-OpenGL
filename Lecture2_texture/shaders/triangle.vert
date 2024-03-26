#version 120 core

attribute vec2 vertexPos;
attribute vec3 vertexColor;

varying vec3 ourColor;
// varying vec4 position;

// uniform float offsetPosition;

void main()
{
    // gl_Position = vec4(vertexPos.x + offsetPosition, vertexPos.y, 0, 1);
    gl_Position = vec4(vertexPos, 0, 1);
    ourColor = vertexColor;
    // position = vec4(vertexPos, 0, 1);
}
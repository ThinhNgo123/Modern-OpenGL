#version 120 core

// varying vec4 position;

varying vec3 ourColor;

uniform float u_deltaTime;

void main()
{
    // gl_FragColor = vec4(ourColor, 1);
    // gl_FragColor = position;
    gl_FragColor = vec4(ourColor.x, u_deltaTime, ourColor.z, 1);
}
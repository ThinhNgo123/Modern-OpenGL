#version 120 core

varying vec3 outColor;
varying vec2 outTexCoord;

uniform sampler2D outTexture;
uniform sampler2D outTexture1;
uniform float mixScale;

void main()
{
    gl_FragColor = mix(
        texture2D(outTexture, outTexCoord), 
        texture2D(outTexture1, vec2(1-outTexCoord.x, outTexCoord.y)),
        mixScale
    );
}
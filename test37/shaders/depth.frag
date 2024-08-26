#version 460 core

in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D texture1;
// uniform sampler2D texture2;
// uniform vec3 objectColor;
// uniform vec3 lightColor;

float near = 0.1, far = 100.0;

void main()
{
    FragColor = texture(texture1, TexCoord);
    // FragColor = vec4(texture(texture1, TexCoord).rgb, 0);
    // vec3 color = texture(texture1, TexCoord).rgb;
    // float z_ndc = 2.0 * gl_FragCoord.z - 1;
    // float z_eye = (-2.0  * near * far) / (z_ndc * (near - far) + near + far);
    // // FragColor = vec4(vec3(1, 1, 0) * vec3((-z_eye + near) / (far + near)), 1);
    // FragColor = vec4(color, 1) * vec4(vec3((-z_eye + near) / (far + near)), 1);
}
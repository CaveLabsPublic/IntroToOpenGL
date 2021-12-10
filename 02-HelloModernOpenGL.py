from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *

import numpy

#
# GLOBALS
vertexDim = 4
nVertices = 3

# Global variable to represent the compiled shader program, written in GLSL
programID = None

# Global variables for buffer objects
VBO = None

# create an array to hold positions of our vertices. numpy array is directly transferable to OpenGL
# order: top-right, bottom-right, bottom-left, top-left
vertexPositions = numpy.array(
	[0.5, 0.5, 0.0, 1.0,
	0.5, -0.5, 0.0, 1.0,
	-0.5, -0.5, 0.0, 1.0],
	dtype='float32'
)

# String containing vertex shader program written in GLSL
strVertexShader = """
#version 330

layout(location = 0) in vec4 vertexPosition;

void main()
{
   gl_Position = vertexPosition;
}
"""

# String containing fragment shader program written in GLSL
strFragmentShader = """
#version 330

out vec4 outColor;

void main()
{
   outColor = vec4(1.0f, 1.0f, 0.0f, 1.0f);
}
"""

#
# FUNCTIONS

# Function that accepts a list of shaders, compiles them, and returns a handle to the compiled program
def createProgram(shaderList):
	programID = glCreateProgram()

	for shader in shaderList:
		glAttachShader(programID, shader)

	glLinkProgram(programID)

	status = glGetProgramiv(programID, GL_LINK_STATUS)
	if status == GL_FALSE:
		strInfoLog = glGetProgramInfoLog(programID)
		print(b"Linker failure: \n" + strInfoLog)

	# important for cleanup
	for shaderID in shaderList:
		glDetachShader(programID, shaderID)

	return programID


# Function that creates and compiles shaders according to the given type (a GL enum value) and
# shader program (a string containing a GLSL program).
def createShader(shaderType, shaderCode):
	shaderID = glCreateShader(shaderType)
	glShaderSource(shaderID, shaderCode)
	glCompileShader(shaderID)

	status = None
	glGetShaderiv(shaderID, GL_COMPILE_STATUS, status)
	if status == GL_FALSE:
		# Note that getting the error log is much simpler in Python than in C/C++
		# and does not require explicit handling of the string buffer
		strInfoLog = glGetShaderInfoLog(shaderID)
		strShaderType = ""
		if shaderType is GL_VERTEX_SHADER:
			strShaderType = "vertex"
		elif shaderType is GL_GEOMETRY_SHADER:
			strShaderType = "geometry"
		elif shaderType is GL_FRAGMENT_SHADER:
			strShaderType = "fragment"

		print(b"Compilation failure for " + strShaderType + b" shader:\n" + strInfoLog)

	return shaderID


# Initialize the OpenGL environment
def init():
	initProgram()
	initVertexBuffer()


# Set up the list of shaders, and call functions to compile them
def initProgram():
	shaderList = []

	shaderList.append(createShader(GL_VERTEX_SHADER, strVertexShader))
	shaderList.append(createShader(GL_FRAGMENT_SHADER, strFragmentShader))

	global programID
	programID = createProgram(shaderList)

	for shader in shaderList:
		glDeleteShader(shader)


# Set up the vertex buffer that will store our vertex coordinates for OpenGL's access
def initVertexBuffer():
	global VBO
	VBO = glGenBuffers(1)

	# set array buffer to our ID
	glBindBuffer(GL_ARRAY_BUFFER, VBO)

	# set data
	bufferData = vertexPositions
	glBufferData( # PyOpenGL allows for the omission of the size parameter
		GL_ARRAY_BUFFER,
		bufferData,
		GL_STATIC_DRAW
	)

	# reset array buffer
	glBindBuffer(GL_ARRAY_BUFFER, 0)


# Called to update the display.
# Because we are using double-buffering, glutSwapBuffers is called at the end
# to write the rendered buffer to the display.
def display():
	glClearColor(0.0, 0.0, 0.0, 0.0)
	glClear(GL_COLOR_BUFFER_BIT)

	# use our program
	glUseProgram(programID)

	# reset our vertex buffer
	glBindBuffer(GL_ARRAY_BUFFER, VBO)
	glEnableVertexAttribArray(0)
	glVertexAttribPointer(0, vertexDim, GL_FLOAT, GL_FALSE, 0, None)

	glDrawArrays(GL_TRIANGLES, 0, nVertices)

	# reset to defaults
	glDisableVertexAttribArray(0)
	glBindBuffer(GL_ARRAY_BUFFER, 0)
	glUseProgram(0)

	glutSwapBuffers()


# keyboard input handler: exits the program if 'esc' is pressed
def keyboard(key, x, y):
	if ord(key) == 27: # ord() is needed to get the keycode
		glutLeaveMainLoop()
		return


# Called whenever the window's size changes (including once when the program starts)
def reshape(w, h):
	glViewport(0, 0, w, h)


# The main function
def main():
	glutInit()
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)

	width = 500;
	height = 500;
	glutInitWindowSize (width, height)

	glutInitWindowPosition (300, 200)

	window = glutCreateWindow("CENG488 Hello Triangle")

	init()
	glutDisplayFunc(display)
	glutReshapeFunc(reshape)
	glutKeyboardFunc(keyboard)

	glutMainLoop();

if __name__ == '__main__':
	main()
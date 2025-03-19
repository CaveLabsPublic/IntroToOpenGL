from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *

from math import sqrt

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
	[3, 3, 0.0, 1.0,
	3, -3, 0.0, 1.0,
	-3, -3, 0.0, 1.0],
	dtype='float32'
)

# String containing vertex shader program written in GLSL
strVertexShader = """
#version 330

layout(location = 0) in vec4 vertexPosition;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

void main()
{
   gl_Position = proj * view * model * vertexPosition;
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

# camera globals
camPosition = numpy.array([0.0, 0.0, 10.0, 1.0], dtype='float32')
camUpAxis = numpy.array([0.0, 1.0, 0.0, 0.0], dtype='float32')
camNear = 1.0
camFar = 100.0
camAspect = 1.0
camFov = 60.0

# objectPosition
objectPosition = numpy.array([0, 0, -5, 1.0], dtype='float32')

#
# FUNCTIONS

# vector stuff
def dot(vec1, vec2):
	return 1.0 * numpy.dot(vec2, vec1)


def cross(vec1, vec2):
	result = numpy.cross(vec1[0:3], vec2[0:3], axisa=0, axisb=0, axisc=0)
	return numpy.array([result[0], result[1], result[2], 0.0], dtype='float32')


def normalize(vec):
	vecLen = sqrt(1.0 * numpy.dot(vec, vec))
	return vec / vecLen


# matrix stuff
def getProjMatrix(near, far, aspect, fov):
	f = numpy.reciprocal(numpy.tan(numpy.divide(numpy.deg2rad(fov), 2.0)))
	base = near - far
	term_0_0 = numpy.divide(f, aspect)
	term_2_2 = numpy.divide(far + near, base)
	term_2_3 = numpy.divide(numpy.multiply(numpy.multiply(2, near), far), base)

	# https://en.wikibooks.org/wiki/GLSL_Programming/Vertex_Transformations
	return  numpy.array([	term_0_0, 0.0, 0.0, 0.0,
							0.0, f, 0.0, 0.0,
							0.0, 0.0, term_2_2, -1,
							0.0, 0.0, term_2_3, 0.0], dtype='float32')


def getViewMatrix():
	# THIS HAS A LOT OF HARD CODED STUFF
	# we first calculate camera x, y, z axises and from those we assemble a rotation matrix.
	# Once that is done we add in the translation.
 	# We assume camera always look at the world space origin.
	# Up vector is always in the direction of global yAxis.
	camZAxis = normalize(numpy.array([-camPosition[0], -camPosition[1], -camPosition[2], 0.0], dtype='float32'))
	camXAxis = cross(camZAxis, camUpAxis)
	camYAxis = cross(camXAxis, camZAxis)

	rotMat = numpy.array([	camXAxis[0], camYAxis[0], -camZAxis[0], 0.0,
							camXAxis[1], camYAxis[1], -camZAxis[1], 0.0,
							camXAxis[2], camYAxis[2], -camZAxis[2], 0.0,
							0.0, 0.0, 0.0, 1.0], dtype='float32').reshape(4,4)

	traMat = numpy.array([	1.0, 0.0, 0.0, 0.0,
							0.0, 1.0, 0.0, 0.0,
							0.0, 0.0, 1.0, 0.0,
							-camPosition[0], -camPosition[1], -camPosition[2], 1.0], dtype='float32').reshape(4,4)

	return traMat.dot(rotMat)


def getModelMatrix():
	return numpy.array([	1.0, 0.0, 0.0, 0.0,
							0.0, 1.0, 0.0, 0.0,
							0.0, 0.0, 1.0, 0.0,
							objectPosition[0], objectPosition[1], objectPosition[2], 1.0], dtype='float32')


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
	# third argument is criptic - in c_types if you multiply a data type with an integer you create an array of that type
	# PyOpenGL allows for the omission of the size parameter
	glBufferData(
		GL_ARRAY_BUFFER,
		(ctypes.c_float * len(vertexPositions))(*vertexPositions),
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

	# get matrices and bind them to vertex shader locations
	modelLocation = glGetUniformLocation( programID, "model" )
	glUniformMatrix4fv(modelLocation, 1, GL_FALSE, getModelMatrix())
	viewLocation = glGetUniformLocation(programID, "view")
	glUniformMatrix4fv(viewLocation, 1, GL_FALSE, getViewMatrix())
	projLocation = glGetUniformLocation(programID, "proj")
	glUniformMatrix4fv(projLocation, 1, GL_FALSE, getProjMatrix(camNear, camFar, camAspect, camFov))

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

	window = glutCreateWindow(b"CENG487 Hello Modern OpenGL")

	init()
	glutDisplayFunc(display)
	glutReshapeFunc(reshape)
	glutKeyboardFunc(keyboard)

	glutMainLoop();

if __name__ == '__main__':
	main()
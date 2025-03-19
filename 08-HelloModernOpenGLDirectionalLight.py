from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *

from math import sqrt
from PIL import Image

import numpy

#
# GLOBALS
vertexDim = 4

# Global variable to represent the compiled shader program, written in GLSL
programID = None

# Global variables for buffer objects
VAO = None

# Global array to hold VBO data
VBOData = None

# Global variable for texture
tex1ID = -1

# create an array to hold positions of our vertices. numpy array is directly transferable to OpenGL
# our vertices has ids like below
#       4---------5
#       |\       /|
#       | 2-----3 |
#       | |     | |
#       | 0-----1 |
#       |/       \|
#       6---------7
vertexPositions = [
	[-1.0, -1.0, 0.0, 1.0],
	[1.0, -1.0, 0.0, 1.0],
	[-1.0, 1.0, 0.0, 1.0],
	[1.0, 1.0, 0.0, 1.0],
	[-2.0, 2.0, -1.0, 1.0],
	[2.0, 2.0, -1.0, 1.0],
	[-2.0, -2.0, -1.0, 1.0],
	[2.0, -2.0, -1.0, 1.0,]
]

# we have 5 faces
# we store indices of the vertices per face
nFaces = 5
nVertices = nFaces * 4
faces = [
	[0, 1, 3, 2],
	[2, 3, 5, 4],
	[6, 7, 1, 0],
	[1, 7, 5, 3],
	[0, 2, 4, 6]
]

# random colors for faces
faceColors = [
	[1.0, 0.0, 0.0, 1.0],
	[0.0, 1.0, 0.0, 1.0],
	[0.0, 0.0, 1.0, 1.0],
	[1.0, 1.0, 0.0, 1.0],
	[0.0, 1.0, 1.0, 1.0]
]

# faces occupy [0,1] space in u and v and they are unfolded like the projection of the shape into XY plane
vertexUVs = [
	[0.33, 0.33],
	[0.66, 0.33],
	[0.33, 0.66],
	[0.66, 0.66],
	[0.0, 1.0],
	[1.0, 1.0],
	[0.0, 0.0],
	[1.0, 0.0]
]

# String containing vertex shader program written in GLSL
strVertexShader = """
#version 330

layout(location = 0) in vec4 vertexPosition;
layout(location = 1) in vec4 vertexColor;
layout(location = 2) in vec2 vertexUV;
layout(location = 3) in vec4 vertexNormal;

out vec4 fragColor;
out vec2 fragUV;
out vec3 fragNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

void main()
{
	gl_Position = proj * view * model * vertexPosition;
	fragColor = vertexColor;
	fragUV = vertexUV;
	fragNormal = normalize( vec3( transpose( inverse(model) ) * vertexNormal ) );
}
"""

# String containing fragment shader program written in GLSL
strFragmentShader = """
#version 330

in vec4 fragColor;
in vec2 fragUV;
in vec3 fragNormal;

out vec4 outColor;

uniform vec3 lightDir;
uniform vec4 lightColor;
uniform float lightIntensity;

uniform sampler2D tex1;

void main()
{
	vec4 texVal = texture(tex1, fragUV);

	// simple lambert diffuse shading model
	float nDotL = max(dot(fragNormal, normalize(lightDir)), 0.0);
	outColor = fragColor * texVal * lightColor * lightIntensity * nDotL;
}
"""

# camera globals
camPosition = numpy.array([5.0, 0.0, 10.0, 1.0], dtype='float32')
camUpAxis = numpy.array([0.0, 1.0, 0.0, 0.0], dtype='float32')
camNear = 1.0
camFar = 100.0
camAspect = 1.0
camFov = 60.0

# objectPosition
objectPosition = numpy.array([0.0, 0.0, 0.0, 1.0], dtype='float32')

# light parameters
lightDir = numpy.array([0.0, 1.0, 1.0, 0.0], dtype='float32')
lightColor =  numpy.array([1.0, 1.0, 1.0, 1.0], dtype='float32')
lightIntensity = 1.0

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
	initVertexBufferData()
	initVertexBuffer()
	initTextures("texture3.png")
	initLightParams()


# Set up the list of shaders, and call functions to compile them
def initProgram():
	shaderList = []

	shaderList.append(createShader(GL_VERTEX_SHADER, strVertexShader))
	shaderList.append(createShader(GL_FRAGMENT_SHADER, strFragmentShader))

	global programID
	programID = createProgram(shaderList)

	for shader in shaderList:
		glDeleteShader(shader)


def initVertexBufferData():
	global VBOData
	global faces

	finalVertexPositions = []
	finalVertexColors = []
	finalVertexUVs = []
	finalVertexNormals = []

	# go over faces and assemble an array for all vertex data
	faceID = 0
	for face in faces:
		# calc a normal for the face
		# we calc the vectors for two edges of the poly and do a cross product to find the normal
		#       | 3-----2 |
		#       | |     | |
		#       | 0-----1 |
		# since our indices are given in counter clockwise order, our two edges are:
		# P1 - P0 and P3 - P0
		edge1 = numpy.array([a - b for a, b in zip(vertexPositions[face[1]], vertexPositions[face[0]])], dtype='float32')
		edge2 = numpy.array([a - b for a, b in zip(vertexPositions[face[3]], vertexPositions[face[0]])], dtype='float32')
		faceNormal = normalize(cross(edge1, edge2))

		# now assemble arrays
		for vertex in face:
			finalVertexPositions.extend(vertexPositions[vertex])
			finalVertexColors.extend(faceColors[faceID])
			finalVertexUVs.extend(vertexUVs[vertex])
			finalVertexNormals.extend(faceNormal)

		faceID += 1

	VBOData = numpy.array(finalVertexPositions + finalVertexColors + finalVertexUVs + finalVertexNormals, dtype='float32')


# Set up the vertex buffer that will store our vertex coordinates for OpenGL's access
def initVertexBuffer():
	global VAO
	global VBOData

	VAO = glGenVertexArrays(1)
	VBO = glGenBuffers(1)

	# bind to our VAO
	glBindVertexArray(VAO)

	# now change the state - it will be recorded in the VAO
	# set array buffer to our ID
	glBindBuffer(GL_ARRAY_BUFFER, VBO)

	# set data
	elementSize = numpy.dtype(numpy.float32).itemsize

	# third argument is criptic - in c_types if you multiply a data type with an integer you create an array of that type
	glBufferData(	GL_ARRAY_BUFFER,
					len(VBOData) * elementSize,
					(ctypes.c_float * len(VBOData))(*VBOData),
					GL_STATIC_DRAW
	)

	# setup vertex attributes
	offset = 0

	# location 0
	glVertexAttribPointer(0, vertexDim, GL_FLOAT, GL_FALSE, elementSize * vertexDim, ctypes.c_void_p(offset))
	glEnableVertexAttribArray(0)

	# define colors which are passed in location 1 - they start after all positions and has four floats consecutively
	offset += elementSize * vertexDim * nVertices
	glVertexAttribPointer(1, vertexDim, GL_FLOAT, GL_FALSE, elementSize * vertexDim, ctypes.c_void_p(offset))
	glEnableVertexAttribArray(1)

	# define uvs which are passed in location 2 - they start after all positions and colors and has two floats per vertex
	offset += elementSize * vertexDim * nVertices
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, elementSize * 2, ctypes.c_void_p(offset))
	glEnableVertexAttribArray(2)

	# define normals which are passed in location 3 - they start after all positions, colors and uvs and has four floats per vertex
	offset += elementSize * 2 * nVertices
	glVertexAttribPointer(3, vertexDim, GL_FLOAT, GL_FALSE, elementSize * vertexDim, ctypes.c_void_p(offset))
	glEnableVertexAttribArray(3)

	# reset array buffers
	glBindBuffer(GL_ARRAY_BUFFER, 0)
	glBindVertexArray(0)


# texture stuff
def initTextures(texFilename):
	# we need to bind to the program to set texture related params
	global programID
	glUseProgram(programID)

	# load texture
	global tex1ID
	tex1ID = loadTexture(texFilename)

	# set shader stuff
	tex1Location = glGetUniformLocation(programID, "tex1")
	glUniform1i(tex1Location, tex1ID)

	# now activate texture units
	glActiveTexture(GL_TEXTURE0 + tex1ID)
	glBindTexture(GL_TEXTURE_2D, tex1ID)

	# reset program
	glUseProgram(0)


def loadTexture(texFilename):
	# load texture - flip vertically to convert from pillow to OpenGL orientation
	image = Image.open(texFilename).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

	# create a new id
	texID = glGenTextures(1)
	# bind to the new id for state
	glBindTexture(GL_TEXTURE_2D, texID)

	# set texture params
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

	# copy texture data
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.size[0], image.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE,
					numpy.frombuffer( image.tobytes(), dtype = numpy.uint8 ) )
	glGenerateMipmap(GL_TEXTURE_2D)

	return texID


def initLightParams():
	# we need to bind to the program to set lighting related params
	global programID
	glUseProgram(programID)

	# set shader stuff
	lightDirLocation = glGetUniformLocation(programID, "lightDir")
	glUniform3f(lightDirLocation, lightDir[0], lightDir[1], lightDir[2])
	lightColorLocation = glGetUniformLocation(programID, "lightColor")
	glUniform4f(lightColorLocation, lightColor[0], lightColor[1], lightColor[2], lightColor[3])
	lightIntensityLocation = glGetUniformLocation(programID, "lightIntensity")
	glUniform1f(lightIntensityLocation, lightIntensity)

	# reset program
	glUseProgram(0)


# Called to update the display.
# Because we are using double-buffering, glutSwapBuffers is called at the end
# to write the rendered buffer to the display.
def display():
	glClearColor(0.0, 0.0, 0.0, 0.0)
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

	# use our program
	glUseProgram(programID)

	# get matrices and bind them to vertex shader locations
	modelLocation = glGetUniformLocation( programID, "model" )
	glUniformMatrix4fv(modelLocation, 1, GL_FALSE, getModelMatrix())
	viewLocation = glGetUniformLocation(programID, "view")
	glUniformMatrix4fv(viewLocation, 1, GL_FALSE, getViewMatrix())
	projLocation = glGetUniformLocation(programID, "proj")
	glUniformMatrix4fv(projLocation, 1, GL_FALSE, getProjMatrix(camNear, camFar, camAspect, camFov))

	# bind to our VAO
	glBindVertexArray(VAO)

	# draw stuff
	glDrawArrays(GL_QUADS, 0, nVertices)

	# reset to defaults
	glBindVertexArray(0)
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

	# need to enable depth testing and depth funct for proper drawing
	glDepthFunc(GL_LEQUAL)
	glEnable(GL_DEPTH_TEST)

	init()
	glutDisplayFunc(display)
	glutReshapeFunc(reshape)
	glutKeyboardFunc(keyboard)

	glutMainLoop();

if __name__ == '__main__':
	main()
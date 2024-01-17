from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *

from math import sqrt
from PIL import Image

import numpy

#
# GLOBALS
windowWidth = 800
windowHeight = 800
vertexDim = 4

# Global variable to represent the compiled shader program, written in GLSL
objectProgramID = None

# we need another shader for depth map rendering
depthMapProgramID = None

# Global variables for buffer objects
VAO_Object = None
VAO_Ground = None

# Global array to hold VBO data
objectVBOData = None
groundVBOData = None

# Global variable for texture
tex1ID = -1

# create two boxes: one as ground to show the shadow and the other as a shadow caster
# our vertices has ids like below
#		  3---------2
#		 /|        /|
#		/ |       / |
# 		0-7------1  6
#       |/       |/
#       4--------5
groundVertexPositions = [
	[-15.0, 0.0, 15.0, 1.0],
	[15.0, 0.0, 15.0, 1.0],
	[15.0, 0.0, -15.0, 1.0],
	[-15.0, 0.0, -15.0, 1.0],
	[-15.0, -0.2, 15.0, 1.0],
	[15.0, -0.2, 15.0, 1.0],
	[15.0, -0.2, -15.0, 1.0],
	[-15.0, -0.2, -15.0, 1.0,]
]

objectVertexPositions = [
	[-1.5, 0.0, 1.5, 1.0],
	[1.5, 0.0, 1.5, 1.0],
	[1.5, 0.0, -1.5, 1.0],
	[-1.5, 0.0, -1.5, 1.0],
	[-1.5, -3.0, 1.5, 1.0],
	[1.5, -3.0, 1.5, 1.0],
	[1.5, -3.0, -1.5, 1.0],
	[-1.5, -3.0, -1.5, 1.0,]
]

# we have 6 faces
# we store indices of the vertices per face
nFaces = 6
nVertices = nFaces * 4
faces = [
	[0, 1, 2, 3],
	[4, 5, 1, 0],
	[5, 6, 2, 1],
	[6, 7, 3, 2],
	[7, 4, 0, 3],
	[7, 6, 5, 4]
]

# all faces are white
faceColors = [
	[1.0, 1.0, 0.0, 1.0],
	[1.0, 0.0, 1.0, 1.0],
	[0.0, 1.0, 0.0, 1.0],
	[0.0, 1.0, 1.0, 1.0],
	[0.0, 0.0, 1.0, 1.0],
	[1.0, 1.0, 0.0, 1.0]
]

# String containing vertex shader program written in GLSL
strObjectVertexShader = """
#version 330

layout(location = 0) in vec4 vertexPosition;
layout(location = 1) in vec4 vertexColor;
layout(location = 2) in vec2 vertexUV;
layout(location = 3) in vec4 vertexNormal;

out vec4 fragColor;
out vec2 fragUV;
out vec3 fragPos;
out vec3 fragNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

void main()
{
	gl_Position = proj * view * model * vertexPosition;
	fragPos = vec3(model * vertexPosition);
	fragColor = vertexColor;
	fragUV = vertexUV;
	fragNormal = normalize( vec3( transpose( inverse(model) ) * vertexNormal ) );
}
"""

# String containing fragment shader program written in GLSL
strObjectFragmentShader = """
#version 330

in vec4 fragColor;
in vec2 fragUV;
in vec3 fragPos;
in vec3 fragNormal;

out vec4 outColor;

uniform vec3 lightPos;
uniform vec3 lightDir;
uniform vec4 lightColor;
uniform float lightIntensity;
uniform float lightCone;
uniform float lightPenumbra;
uniform mat4 lightView;
uniform mat4 lightProj;
uniform sampler2D depthMapTex;

uniform sampler2D tex1;

float calcShadow(vec4 fragPosLightSpace)
{
	// perform perspective divide to go to NDC
	vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;

	// Transform to [0,1] range
	projCoords = projCoords * 0.5 + 0.5;

	// Get depth of current fragment from light's perspective
	float currentDepth = projCoords.z;

	// PCF - percentage closer filtering
	// sample shadowmap multiple times at slightly offset positions
	// and take average
	float shadow = 0.0;
	vec2 texelSize = 1.0 / textureSize(depthMapTex, 0);
	int samples = 0;
	for(int x = -2; x <= 2; ++x)
	{
		for(int y = -2; y <= 2; ++y)
		{
			float pcfDepth = texture(depthMapTex, projCoords.xy + vec2(x, y) * texelSize).r;
			shadow += currentDepth > pcfDepth  ? 1.0 : 0.0;
			samples += 1;
		}
	}
	shadow /= float(samples);

	// Keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
	if(projCoords.z > 2.0) {
		shadow = 0.0;
	}

	// fake ambient light by scaling shadows
	return shadow * 0.8;
}

void main()
{
	vec4 texVal = texture(tex1, fragUV);

	// simple lambert diffuse shading model

	// calc falloff for spotlight
	float falloff = 1.0;
	vec3 lightVector = normalize(lightPos - fragPos);
	float fragAngleCos = dot(-lightVector, normalize(lightDir));
	float coneCos = cos(radians(lightCone));
	float penumbraCos = cos(radians(lightPenumbra + lightCone));
	if ( fragAngleCos >= coneCos ) {
		falloff = 1.0;
	} else if ( fragAngleCos <= penumbraCos ) {
		falloff = 0.0;
	} else {
		falloff = ( fragAngleCos - penumbraCos ) / (coneCos - penumbraCos);
	}
	// fake ambient light by scaling falloff
	falloff = (1.0 - (1.0 - falloff) * 0.9);

	// calc shadow
	vec4 fragPosLightSpace = lightProj * lightView * vec4( fragPos.x, fragPos.y, fragPos.z, 1.0 );
	float shadow = calcShadow(fragPosLightSpace);

	float nDotL = max(dot(fragNormal, lightVector), 0.0);
	outColor = fragColor * texVal * lightColor * lightIntensity * falloff * (1.0 - shadow) * nDotL;
}
"""

strDepthMapVertexShader = """
#version 330 core
layout (location = 0) in vec4 vertexPosition;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

void main()
{
	gl_Position = proj * view * model * vertexPosition;
}
"""

strDepthMapFragmentShader = """
#version 330 core

void main()
{
   // color = outColor * outIntensity;
}
"""

# camera globals
camPosition = numpy.array([10.0, 10.0, 20.0, 1.0], dtype='float32')
camDirection = numpy.array([-camPosition[0], -camPosition[1], -camPosition[2], 0.0], dtype='float32')
camUpAxis = numpy.array([0.0, 1.0, 0.0, 0.0], dtype='float32')
camNear = 1.0
camFar = 100.0
camAspect = 1.0
camFov = 60.0

# objectPosition
objectPosition = numpy.array([5.0, 7.0, 5.0, 1.0], dtype='float32')
groundPosition = numpy.array([0.0, 0.0, 0.0, 1.0], dtype='float32')

# light parameters
lightPos = numpy.array([10.0, 10.0, 8.0, 1.0], dtype='float32')
lightDir = numpy.array([-1.0, -1.2, -0.5, 1.0], dtype='float32')
lightColor =  numpy.array([1.0, 1.0, 1.0, 1.0], dtype='float32')
lightIntensity = 1.5
lightCone = 20
lightPenumbra = 30
lightCamAspect = 1.0
lightCamFov = 2.0 * (lightCone + lightPenumbra)
lightCamNear = 1.0
lightCamFar = 1000.0
lightUpAxis = numpy.array([0.0, 1.0, 0.0, 0.0], dtype='float32')

# depth map parameters
depthMapTex = None
depthMapBuffer = None
depthMapRes = 2048
depthMapRendered = False

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


def getViewMatrix(position, direction, upAxis):
	# THIS HAS A LOT OF HARD CODED STUFF
	# we first calculate camera x, y, z axises and from those we assemble a rotation matrix.
	# Once that is done we add in the translation.
 	# In case of a regular camera we assume camera always look at the world space origin.
	# In case of a light camera, we map the direction vector to z axis
	camZAxis = normalize(direction)
	camXAxis = cross(camZAxis, upAxis)
	camYAxis = cross(camXAxis, camZAxis)

	rotMat = numpy.array([	camXAxis[0], camYAxis[0], -camZAxis[0], 0.0,
							camXAxis[1], camYAxis[1], -camZAxis[1], 0.0,
							camXAxis[2], camYAxis[2], -camZAxis[2], 0.0,
							0.0, 0.0, 0.0, 1.0], dtype='float32').reshape(4,4)

	traMat = numpy.array([	1.0, 0.0, 0.0, 0.0,
							0.0, 1.0, 0.0, 0.0,
							0.0, 0.0, 1.0, 0.0,
							-position[0], -position[1], -position[2], 1.0], dtype='float32').reshape(4,4)

	return traMat.dot(rotMat)


def getModelMatrix(location):
	return numpy.array([	1.0, 0.0, 0.0, 0.0,
							0.0, 1.0, 0.0, 0.0,
							0.0, 0.0, 1.0, 0.0,
							location[0], location[1], location[2], 1.0], dtype='float32')


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
	else:
		print("\tCompiled shader!")

	return shaderID


# Initialize the OpenGL environment
def init():
	global objectProgramID, depthMapProgramID
	global objectVBOData, groundVBOData
	global VAO_Object, VAO_Ground

	objectProgramID = initProgram(strObjectVertexShader, strObjectFragmentShader, "Object Shaders")
	objectVBOData = initVertexBufferData(objectVertexPositions, faces, faceColors)
	groundVBOData = initVertexBufferData(groundVertexPositions, faces, faceColors)
	VAO_Object = initVertexBuffer(objectVBOData, nVertices)
	VAO_Ground = initVertexBuffer(groundVBOData, nVertices)

	initTextures(objectProgramID, "texture4.png")

	depthMapProgramID = initProgram(strDepthMapVertexShader, strDepthMapFragmentShader, "Depth Map Shaders")
	initDepthMapBuffer()

	initLightParams(objectProgramID)


# Set up the list of shaders, and call functions to compile them
def initProgram(vertexShader, fragmentShader, msg):
	shaderList = []

	print("Creating " + msg + "...")
	shaderList.append(createShader(GL_VERTEX_SHADER, vertexShader))
	shaderList.append(createShader(GL_FRAGMENT_SHADER, fragmentShader))

	programID = createProgram(shaderList)

	for shader in shaderList:
		glDeleteShader(shader)

	return programID


def initVertexBufferData(vertexPositions, faces, faceColors):
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
		uvIndex = 0
		for vertex in face:
			finalVertexPositions.extend(vertexPositions[vertex])
			finalVertexColors.extend(faceColors[faceID])

			# this assumes a quad and fully uses the uv range per face
			if uvIndex == 0:
				finalVertexUVs.extend ( [ 0.0, 0.0 ] )
			elif uvIndex == 1:
				finalVertexUVs.extend ( [ 1.0, 0.0 ] )
			elif uvIndex == 2:
				finalVertexUVs.extend ( [ 1.0, 1.0 ] )
			else:
				finalVertexUVs.extend ( [ 0.0, 1.0 ] )

			# finalVertexUVs.extend(vertexUVs[vertex])
			finalVertexNormals.extend(faceNormal)
			uvIndex += 1

		faceID += 1

	return numpy.array(finalVertexPositions + finalVertexColors + finalVertexUVs + finalVertexNormals, dtype='float32')


# Set up the vertex buffer that will store our vertex coordinates for OpenGL's access
def initVertexBuffer(VBOData, nVertices):
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

	return VAO


# texture stuff
def initTextures(programID, texFilename):
	# we need to bind to the program to set texture related params
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
	# load texture - flip int verticallt to convert from pillow to OpenGL orientation
	image = Image.open(texFilename).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

	# create a new id
	texID = glGenTextures(1)
	# bind to the new id for state
	glBindTexture(GL_TEXTURE_2D, texID)

	# set texture params
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

	# copy texture data
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.size[0], image.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE,
					numpy.frombuffer( image.tobytes(), dtype = numpy.uint8 ) )
	glGenerateMipmap(GL_TEXTURE_2D)

	return texID


def initLightParams(programID):
	# we need to bind to the program to set lighting related params
	glUseProgram(programID)

	# set shader stuff
	lightPosLocation = glGetUniformLocation(programID, "lightPos")
	glUniform3f(lightPosLocation, lightPos[0], lightPos[1], lightPos[2])
	lightDirLocation = glGetUniformLocation(programID, "lightDir")
	glUniform3f(lightDirLocation, lightDir[0], lightDir[1], lightDir[2])
	lightColorLocation = glGetUniformLocation(programID, "lightColor")
	glUniform4f(lightColorLocation, lightColor[0], lightColor[1], lightColor[2], lightColor[3])
	lightIntensityLocation = glGetUniformLocation(programID, "lightIntensity")
	glUniform1f(lightIntensityLocation, lightIntensity)
	lightConeLocation = glGetUniformLocation(programID, "lightCone")
	glUniform1f(lightConeLocation, lightCone)
	lightPenumbraLocation = glGetUniformLocation(programID, "lightPenumbra")
	glUniform1f(lightPenumbraLocation, lightPenumbra)
	lightViewLocation = glGetUniformLocation( programID, "lightView" )
	glUniformMatrix4fv(lightViewLocation, 1, GL_FALSE, getViewMatrix(lightPos, lightDir, lightUpAxis))
	lightProjLocation = glGetUniformLocation( programID, "lightProj" )
	glUniformMatrix4fv(lightProjLocation, 1, GL_FALSE, getProjMatrix(lightCamNear, lightCamFar, lightCamAspect, lightCamFov))
	depthMapTexLocation = glGetUniformLocation( programID, "depthMapTex" )
	glUniform1i(depthMapTexLocation, depthMapTex)

	# now activate texture unit
	glActiveTexture(GL_TEXTURE0 + depthMapTex)
	glBindTexture(GL_TEXTURE_2D, depthMapTex)

	# reset program
	glUseProgram(0)


def initDepthMapBuffer():
	global depthMapTex, depthMapBuffer, depthMapRes

	depthMapTex = glGenTextures(1)
	glActiveTexture(GL_TEXTURE0 + depthMapTex)
	glBindTexture(GL_TEXTURE_2D, depthMapTex)
	glTexImage2D( 	GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
					depthMapRes, depthMapRes, 0,
					GL_DEPTH_COMPONENT, GL_FLOAT, None )
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
	# set border color to 1 which means no shadow since it is the max depth value
	glTexParameterfv( GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [1.0, 1.0, 1.0, 1.0] )

	# create a framebuffer
	depthMapBuffer = glGenFramebuffers(1)
	glBindFramebuffer(GL_FRAMEBUFFER, depthMapBuffer)
	# bind texture to framebuffers depth attachment
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMapTex, 0)

	# tell OpenGL we dont need color buffer as otherwise framebuffer would be incomplete
	glDrawBuffer(GL_NONE)
	glReadBuffer(GL_NONE) # to avoid problems with some GPUS supporting only OpenGL3.x

	# unbind buffers for cleanup
	glBindTexture(GL_TEXTURE_2D, 0)  # unbind texture
	glActiveTexture(GL_TEXTURE0)  # set active TU to 0 again

	glBindFramebuffer(GL_FRAMEBUFFER, 0)  # unbind framebuffer


# Called to update the display.
# Because we are using double-buffering, glutSwapBuffers is called at the end
# to write the rendered buffer to the display.
def display():
	# render depth map
	renderDepthMap()

	# now reset viewport and do proper render
	glViewport(0, 0, windowWidth, windowHeight)
	glClearColor(0.0, 0.0, 0.0, 0.0)
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

	# draw objects
	viewMatrix = getViewMatrix(camPosition, camDirection, camUpAxis)
	projMatrix = getProjMatrix(camNear, camFar, camAspect, camFov)
	draw(objectProgramID, VAO_Object, nVertices, objectPosition, viewMatrix, projMatrix)
	draw(objectProgramID, VAO_Ground, nVertices, groundPosition, viewMatrix, projMatrix)

	glutSwapBuffers()


# called to draw an object
def draw(programID, VAO, nVertices, location, viewMatrix, projMatrix):
	# use our program
	glUseProgram(programID)

	# get matrices and bind them to vertex shader locations
	modelLocation = glGetUniformLocation( programID, "model" )
	glUniformMatrix4fv(modelLocation, 1, GL_FALSE, getModelMatrix(location))
	viewLocation = glGetUniformLocation(programID, "view")
	glUniformMatrix4fv(viewLocation, 1, GL_FALSE, viewMatrix)
	projLocation = glGetUniformLocation(programID, "proj")
	glUniformMatrix4fv(projLocation, 1, GL_FALSE, projMatrix)

	# bind to our VAO
	glBindVertexArray(VAO)

	# draw stuff
	glDrawArrays(GL_QUADS, 0, nVertices)

	# reset to defaults
	glBindVertexArray(0)
	glUseProgram(0)


def renderDepthMap():
	global depthMapRendered, depthMapBuffer, depthMapRes, depthMapTex

	if not depthMapRendered:
		print("Rendering Depth Map Shadows...")

		# configureshader
		glUseProgram(depthMapProgramID)

		viewMatrix = getViewMatrix(lightPos, lightDir, lightUpAxis)
		projMatrix = getProjMatrix(lightCamNear, lightCamFar, lightCamAspect, lightCamFov)

		viewLocation = glGetUniformLocation(depthMapProgramID, "view")
		projLocation = glGetUniformLocation(depthMapProgramID, "proj")

		# set matrices for view and proj
		glUniformMatrix4fv(viewLocation, 1, GL_FALSE, viewMatrix)
		glUniformMatrix4fv(projLocation, 1, GL_FALSE, projMatrix)

		# setup viewport
		glClearColor(0.0, 0.0, 0.0, 0.0)
		glViewport(0, 0, depthMapRes, depthMapRes)
		glBindFramebuffer(GL_FRAMEBUFFER, depthMapBuffer )
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

		# cull front faces to improve peter panning!
		glEnable(GL_CULL_FACE)
		glCullFace(GL_FRONT)

		# render
		draw(depthMapProgramID, VAO_Object, nVertices, objectPosition, viewMatrix, projMatrix)
		draw(depthMapProgramID, VAO_Ground, nVertices, groundPosition, viewMatrix, projMatrix)

		# unbind framebuffer
		glBindFramebuffer(GL_FRAMEBUFFER, 0 )
		glCullFace(GL_BACK) # reset to default
		glDisable(GL_CULL_FACE) # reset to default

		depthMapRendered = True
		print("\tDone")


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

	glutInitWindowSize (windowWidth, windowHeight)

	glutInitWindowPosition (300, 100)

	window = glutCreateWindow("CENG487 Hello Depth Map")

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
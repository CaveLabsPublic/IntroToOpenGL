from math import pi,sin,cos,sqrt,acos
from vector import *
import numpy

class Matrix:
	@staticmethod
	def create(arg):
		if len(arg) != 16:
			raise Exception("array size must be 16!")

		return Matrix( arg )

	def __init__(self, rows = None):
		if rows is None:
			self.na = numpy.identity(4)
		else:
			self.na = numpy.array( rows )
			self.na = numpy.reshape( self.na, (4, 4) )
			self.na = numpy.transpose( self.na )

	def fromNumpy(self, a):
		self.na = a

	def __str__(self):
		string = ""
		for x in self.na:
			string += str(x) + "\n"
		return string

	def asList(self):
		return self.na.ravel().tolist()

	def asNumpy(self):
		return self.na

	def transpose(self):
		return Matrix( numpy.transpose( self.na ) )

	def rowsize(self):
		return 4

	def colsize(self):
		return 4

	def vecmul(self, vector):
		result = vector.na.reshape( (1, 4) ).dot( self.na ).ravel()
		return Vector3f( result[0], result[1], result[2] )

	def pointmul(self, vector):
		result = vector.na.reshape( (1, 4) ).dot( self.na ).ravel()
		return Point3f( result[0], result[1], result[2] )

	def product(self, other):
		result = Matrix()
		result.fromNumpy( self.na.dot( other.na ) )
		#result.na = result.na.transpose()
		return result

	@staticmethod
	def product3(mat1, mat2, mat3):
		tmp = mat3.product(mat2)
		return tmp.product(mat1)

	def __add__(self, other):
		return Matrix( self.na + other.na )

	def __mul__(self, scalar):
		return Matrix( scalar * self.na )

	def __rmul__(self, scalar):
		return self.__mul__(scalar)

	@staticmethod
	def Rx(x):
		return Matrix.create( [1.0, 0.0, 0.0, 0.0, 0.0, cos(x), -sin(x), 0.0, 0.0, sin(x), cos(x), 0.0, 0.0, 0.0, 0.0, 1.0] )

	@staticmethod
	def Ry(x):
		return Matrix.create( [cos(x), 0.0, sin(x), 0.0, 0.0, 1.0, 0.0, 0.0,-sin(x), 0.0, cos(x), 0.0, 0.0, 0.0, 0.0, 1.0] )

	@staticmethod
	def Rz(x):
		return Matrix.create( [cos(x), -sin(x), 0.0, 0.0, sin(x), cos(x), 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0] )

	@staticmethod
	def S(scalar):
		return Matrix.create( [scalar, 0.0, 0.0, 0.0, 0.0, scalar, 0.0, 0.0, 0.0, 0.0, scalar, 0.0, 0.0, 0.0, 0.0, 1.0] )

	@staticmethod
	def T(x,y,z):
		return Matrix.create( [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z, 0.0, 0.0, 0.0, 1.0] )

	@staticmethod
	def identity():
		return Matrix.create( [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0] )

	@staticmethod
	def zeros():
		return Matrix.create( [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] )



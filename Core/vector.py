import numpy
from math import pi,sin,cos,sqrt,acos

__all__ = ['HCoord', 'Vector3f', 'Point3f', 'ColorRGBA']

class HCoord:
	def __init__(self, a):
		self.na = a

	def from4d(self, x, y, z, w):
		self.na = numpy.array( [x, y, z, w] )

	def x(self):
		return self.na[0]

	def y(self):
		return self.na[1]

	def z(self):
		return self.na[2]

	def w(self):
		return self.na[3]

	def sqrlen(self):
		return 1.0 * numpy.dot ( self.na, self.na )

	def len(self):
		return sqrt( self.sqrlen() )

	def dot(self, other):
		return 1.0 * numpy.dot(other.na, self.na)

	def cross(self, other):
		result = numpy.cross( self.na[0:3], other.na[0:3], axisa = 0, axisb = 0, axisc = 0 )
		return Vector3f( result[0], result[1], result[2] )

	def cosa(self, other):
		return min( max( self.dot(other) / (self.len() * other.len() ), 0.0), 1.0 )

	def angle(self, other):
		return acos(self.cosa(other))

	def normalize(self):
		vecLen = self.len()
		self.na = self.na / vecLen
		return self

	def project(self, other):
		return other.normalize() * (self.len() * self.cosa(other))

	def Rx(self,x):
		m = Matrix.create( [1, 0, 0, 0, 0, cos(x), -sin(x), 0, 0, sin(x), cos(x), 0, 0, 0, 0, 1] )
		return m.vecmul(self)

	def Ry(self, x):
		m = Matrix.create( [cos(x), 0, sin(x), 0, 0, 1, 0,0, -sin(x), 0, cos(x), 0, 0, 0, 0, 1] )
		return m.vecmul(self)

	def Rz(self, x):
		m = Matrix.create([cos(x), -sin(x), 0, 0, sin(x), cos(x), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
		return m.vecmul(self)

	def S(self,scalar):
		return self.na * scalar

	def T(self,x,y,z):
		m = Matrix.create([ 1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1] )
		return m.vecmul(self)

	def __add__(self, other):
		self.na = self.na + other.na
		return self

	def __sub__(self, other):
		self.na = self.na - other.na
		return self

	def __div__(self, scalar):
		if (scalar == 0):
			return self
		else:
			self.na = self.na / scalar
			return self

	def __mul__(self, scalar):
		self.na = scalar * self.na
		return self

	def __rmul__(self, scalar):
		self.na = scalar * self.na
		return self

	def __str__(self):
		return "(" + str( self.x() ) + " " + str( self.y() ) + " " + str( self.z() ) + " " + str( self.w() ) +")"

	def __repr__(self):
		return self.__str__()

class Vector3f(HCoord):
	def __init__(self,x,y,z):
		self.from4d(x, y, z, 0.0)

	def fromNumpy(self, na):
		self.na = na


class Point3f(HCoord):
	def __init__(self, x, y, z):
		self.from4d(x, y, z, 1.0)

	def __sub__(self, other):
		result = self.na - other.na
		return Vector3f( result[0], result[1], result[2] )

	def __add__(self, other):
		result = self.na + other.na
		return Point3f( result[0], result[1], result[2] )

class ColorRGBA(HCoord):
	def __init__(self, r, g, b, a):
		self.from4d(r, g, b, a)

	def r(self):
		return self.na[0]

	def g(self):
		return self.na[1]

	def b(self):
		return self.na[2]

	def a(self):
		return self.na[3]
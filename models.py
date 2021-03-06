from server import db

###
#Glyph model definitions for font files
###

class Font(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	name = db.Column(db.String(128))
	family = db.Column(db.String())
	style = db.Column(db.String(32))
	
	ascent = db.Column(db.Integer)
	descent = db.Column(db.Integer)
	units_per_em = db.Column(db.Integer)

	xMin = db.Column(db.Integer)
	xMax = db.Column(db.Integer)
	yMin = db.Column(db.Integer)
	yMax = db.Column(db.Integer)

	glyphs = db.relationship('Glyph', back_populates='font', cascade="all, delete-orphan")
	def __repr__(self):
		return '<Font {}>'.format(self.family + " " + self.style)

class UnicodeBlock(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	name = db.Column(db.String(64))
	start = db.Column(db.Integer)
	end = db.Column(db.Integer)
	glyphs = db.relationship('Glyph', back_populates='block')

composite_table = db.Table('composite', 
	db.Column('glyph_id', db.Integer, db.ForeignKey('glyph.id')),
	db.Column('contour_id', db.Integer, db.ForeignKey('contour.id')))

class Glyph(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	unicode = db.Column(db.Integer)
	name = db.Column(db.String(64))

	advance_width = db.Column(db.Integer)
	left_side_bearing = db.Column(db.Integer)
	#right_side_bearing = advance_width - (left_side_bearing + xMax - xMin)
	xMin = db.Column(db.Integer)
	xMax = db.Column(db.Integer)
	yMin = db.Column(db.Integer)
	yMax = db.Column(db.Integer)

	simple = db.Column(db.Boolean)

	#These relationships are sort of mutually exclusive:
	#Offsets only appear on a glyph when its composite i.e. not simple. Offsets point to contours on simple glyphs
	#Note: Must explicitly specify the foreign key on the Offset table since it contains two foreign keys pointing back to the Glyph table
	offsets = db.relationship('Offset', back_populates='composite_glyph', cascade="all, delete-orphan")
	
	#Simple glyphs point directly to their contours
	#contours = db.relationship('Contour', back_populates='glyph', cascade="all, delete-orphan")
	contours = db.relationship('Contour', back_populates='glyphs', secondary=composite_table)

	#If you have a two-part model name, use underscore in the stringification
	block_id = db.Column(db.Integer, db.ForeignKey('unicode_block.id'))
	block = db.relationship('UnicodeBlock', back_populates='glyphs')

	font_id = db.Column(db.Integer, db.ForeignKey('font.id'))
	font = db.relationship('Font', back_populates="glyphs")
	
	def __repr__(self):
		return '<Glyph {}>'.format(self.id)

#We need a way to link offsets to lists of contours without making glyph records
class Offset(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	x = db.Column(db.Integer)
	y = db.Column(db.Integer)

	glyph_name = db.Column(db.String(64))

	composite_id = db.Column(db.Integer, db.ForeignKey('glyph.id'))
	composite_glyph = db.relationship('Glyph', back_populates='offsets') 

class Contour(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	orientation = db.Column(db.Boolean)
	strokes = db.relationship('Stroke', back_populates='contour', cascade="all, delete-orphan")
	
	glyphs = db.relationship('Glyph', back_populates="contours", secondary=composite_table)

	drawing_id = db.Column(db.Integer, db.ForeignKey('drawing.id'))
	drawing = db.relationship('Drawing', back_populates="contours")

class Stroke(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	type = db.Column(db.String(1))
	order = db.Column(db.Integer)
	
	#point_id = db.Column(db.Integer, db.ForeignKey('point.id'))
	point = db.relationship('Point', back_populates='stroke', uselist=False, cascade="all, delete-orphan")
	
	contour_id = db.Column(db.Integer, db.ForeignKey('contour.id'))
	contour = db.relationship('Contour', back_populates='strokes')

class Point(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	x = db.Column(db.Float)
	y = db.Column(db.Float)
	
	stroke_id = db.Column(db.Integer, db.ForeignKey('stroke.id'))
	stroke = db.relationship('Stroke', back_populates='point')

###
#Quick, Draw! representation:
#Drawing contains contours (which they call strokes)
#Each stroke is an array of x-y coordinates - we parse these as many
#strokes, each automatically an 'L' type, since they simplified the data
#into a series of vector lines.
###
class Drawing(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	word = db.Column(db.String(64))
	country_code = db.Column(db.String(2))
	timestamp = db.Column(db.String(32))
	recognized = db.Column(db.Boolean)

	contours = db.relationship('Contour', back_populates='drawing', cascade="all, delete-orphan")

###
#Massive text storage to create the voices of characters
###
#Top-level abstraction is a 'writing' - like a religious text, novel or publication
#class Writing(db.Model):
#	id = db.Column(db.Integer, primary_key=True)
#	text = db.Column(db.Text)

#More difficult questions posed by lower-level representations. Chapters?
#We need to add some kind of rich metadata in order to create useful datasets
#for training text generators as character voices
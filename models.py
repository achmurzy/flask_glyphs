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

class UnicodeBlock(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	name = db.Column(db.String(64))
	start = db.Column(db.Integer)
	end = db.Column(db.Integer)
	glyphs = db.relationship('Glyph', back_populates='block')

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

	#These relationships are sort of mutually exclusive
	offsets = db.relationship('Offset', back_populates='glyph', cascade="all, delete-orphan")
	contours = db.relationship('Contour', back_populates='glyph', cascade="all, delete-orphan")

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

	glyph_id = db.Column(db.Integer, db.ForeignKey('glyph.id'))
	glyph = db.relationship('Glyph', back_populates="offsets")

	composite_name = db.Column(db.String(64))
	composite_glyph = db.relationship('Glyph') 

class Contour(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	strokes = db.relationship('Stroke', back_populates='contour', cascade="all, delete-orphan")
	
	glyph_id = db.Column(db.Integer, db.ForeignKey('glyph.id'))
	glyph = db.relationship('Glyph', back_populates="contours")
	
	drawing_id = db.Column(db.Integer, db.ForeignKey('drawing.id'))
	drawing = db.relationship('Drawing', back_populates="contours")

class Stroke(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	type = db.Column(db.String(1))
	order = db.Column(db.Integer)
	points = db.relationship('Point', back_populates='strokes', cascade="all, delete-orphan")
	
	contour_id = db.Column(db.Integer, db.ForeignKey('contour.id'))
	contour = db.relationship('Contour', back_populates='strokes')

class Point(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	x = db.Column(db.Float)
	y = db.Column(db.Float)
	
	stroke_id = db.Column(db.Integer, db.ForeignKey('stroke.id'))
	strokes = db.relationship('Stroke', back_populates='points')

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
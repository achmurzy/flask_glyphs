from server import db
###
#Glyph model definitions
#Brad Boyle would likely write this type of thing in bash and psql
#Using an ORM, we can write DB code in python3. Very nice
###

#Training sets will be composed of one or more alphabets

#We may prefer a many-to-many relationship here, so that we can
#assign the same glyph to multiple alphabets.
class Alphabet(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	glyphs = db.relationship('Glyph', backref='alphabet', lazy='dynamic')

class Glyph(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	#Glyphs are nothing more than a list of strokes. 
	strokes = db.relationship('Stroke', backref='glyph', lazy='dynamic')
	
	def __repr__(self):
		return '<Glyph {}>'.format(self.id)

class Stroke(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	type = db.Column(db.String(1))
	points = db.relationship('Point', backref='stroke', lazy='dynamic')
	glyph_id = db.Column(db.Integer, db.ForeignKey('glyph.id'))

class Point(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	x = db.Column(db.Float)
	y = db.Column(db.Float)
	stroke_id = db.Column(db.Integer, db.ForeignKey('stroke.id'))
	line_id = db.Column(db.Integer, db.ForeignKey('line.id'))

###
#Drawing processing
###
class Drawing(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	word = db.Column(db.String(64))
	country_code = db.Column(db.String(2))
	timestamp = db.Column(db.String(32))
	recognized = db.Column(db.Boolean)
	drawing = db.relationship('Line', backref='drawing', lazy='dynamic')

#This probably isn't going to work because the Point model isn't
#expecting to be associated with a Line. Could add line_id to model
class Line(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	order = db.Column(db.Integer)
	points = db.relationship('Point', backref='line', lazy='dynamic')
	drawing_id = db.Column(db.Integer, db.ForeignKey('drawing.id'))
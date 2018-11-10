from server import db

class Glyph(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	#Glyphs are nothing more than a list of strokes. 
	#How do we represent hierarchical models in a convenient way?

	def __repr__(self):
		return '<Glyph {}>'.format(self.id)
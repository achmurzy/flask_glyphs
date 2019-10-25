from models import *
from server import db
# Functions for querying the glyph database in interesting ways
# e.g. per Unicode block, or according to a particularly style (e.g. Serif, Sans-Serif, etc.)

#Takes the name of a unicode block (defined in unicode_blocks.txt)
#returns all glyphs in the database with unicode mapping falling in that range
def get_unicode_block_glyphs(block_name):
	block = db.session.query(UnicodeBlock).filter(UnicodeBlock.name == block_name).first()
	glyphs = db.session.query(Glyph).filter((Glyph.unicode >= block.start) & (Glyph.unicode <= block.end)).all()
	return(glyphs)

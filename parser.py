import ndjson
from server import db
from models import Drawing, Line, Point

def get_ndjson(file):
	with open(file) as f:
		data = ndjson.load(f)
		return data

def store_ndjson(json):
	for drawing in json:
		pic = Drawing(id=drawing['key_id'], word=drawing['word'], 
country_code=drawing['countrycode'], timestamp=drawing['timestamp'],
recognized=drawing['recognized'], drawing=parse_drawing(drawing))
		db.session.add(pic)

def parse_drawing(drawing):
	lines = []
	for index, line in enumerate(drawing['drawing']):
		points = []
		length = len(line[0])
		for i in range(length):
			pp = Point(x=line[0][i], y=line[1][i])
			points.append(pp)
			db.session.add(pp)
		line = Line(order=index, drawing_id=drawing['key_id'])#, points = points)
		db.session.add(line)
		lines.append(line)
	return lines


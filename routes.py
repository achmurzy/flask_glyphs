from models import Font, Glyph, Stroke, Point
from server import app, db
from encoder import recursive_alchemy_encoder

from flask import jsonify, request
import requests, json

global_glyph_data = 'empty'
#Store a full window's worth of glyphs
# - send a set of training data to the server for processing
@app.route('/store_glyph', methods = ['POST'])
def store_glyph():
	global global_glyph_data
	global_glyph_data = request.get_json();
	#store_glyphs(global_glyph_data)
	#session.add(glyph)
	return 'OK'

#Simple method to render a full-window's worth of glyphs
# - or lets say serve a set of generated glyphs
@app.route('/get_glyph', methods = ['GET'])
def send_glyph():
	prediction_font = db.session.query(Font).filter(Font.name == "Predictions").all()[-1]	#Get the most recent predictions
	glyph_data = prediction_font.glyphs
	response = app.response_class(response=json.dumps(glyph_data, cls=recursive_alchemy_encoder(True, ['name', 'unicode', 'advance_width', 'contours', 'orientation', 'strokes', 'order', 'type', 'point', 'x', 'y']), check_circular=False), status=200, mimetype='application/json')
	return response
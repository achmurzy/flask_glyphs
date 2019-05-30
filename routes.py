from models import Glyph, Stroke, Point
from server import app
from flask import jsonify, request
import requests

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
	print(global_glyph_data)
	return jsonify(global_glyph_data)
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sqlalchemy.dialects import postgresql

from models import Glyph, Font

#Our sequential data is hierarchical:
#-Sequences of fonts/languages
#-Sequences of glyphs
#-Sequences of contours
#-Sequences of strokes
#Only the latter of these is truly sequential, and the concept of 'samples'
#should exist as the set of strokes on a contour. Thus, we're modelling contours
#rather than glyphs. How to assemble contours into glyphs might require
#annotating transitions between glyphs in the sequence, just like we annotate
#transitions between contours using the 'M' symbol in strokes. 
#Problems of this kind typically learn offsets between points rather than the actual positions of the points themselves.

#Number of contours per tensor
batch_size = 1

#Number of strokes (computed per-batch) in a contour, our sequence length
sequence_length = 0

#If we ever really learn SQL we could potentially draw directly from the db.
#query_string = query.statement.compile(dialect=postgresql.dialect())
#tf.data.experimental.SqlDataset('sqlite', 'app.db', query_string, (tf.float32))

#Deconstruct a Glyph parse into a list of points on all the glyphs. As an example, these are all from arabtype.ttf
font = db.session.query(Font).filter(Font.name == 'arabtype.ttf').first()

#Goal of our input vector = [batch_size, sequence_length, xy], where xy = 2
query = db.session.query(Glyph).limit(5)
stroke_tensor = []
contour_tensor = []
for glyph in query.all():
	for contour in glyph.contours:
		for stroke in contour.strokes:
			for point in stroke.points:
				stroke_tensor.append([point.x, point.y])
		contour_tensor.append(stroke_tensor)
		stroke_tensor = []
	#We aren't curently including any demarcations between glyphs, which would be another dimension in our sequential dataset

#Used with Dataset.map() to normalize according to metadata on the source font.
def normalize_font_glyphs(dataset, font):
	points = []
	for point in dataset:
		x = (point[0] - font.xMin) / (font.xMax - font.xMin)
		y = (point[1] - font.yMin) / (font.yMax - font.yMin)
		points.append([x,y])
	return points

def pad_batch(normalized, seq_length):
	for stroke in normalized:
		diff = seq_length - len(stroke)
		stroke.extend([[0,0]] * diff)
	return normalized


#Batch and normalize the data, while computing the maximum sequence length per-batch and padding accordingly
#Can't use this because keras doesn't support variable-length sequences in this way.
def padded_normalized_manual(contour_tensor):
	normalized_batch = []
	counter = 0
	while counter < len(contour_tensor):
		batch = counter
		max_seq = 0
		normalized = []
		for i in range(batch, batch+batch_size):
			if(i >= len(contour_tensor)):
				remain = i % batch_size
				for j in range(remain, batch_size):
					normalized.append([[0,0]] * max_seq)
				break
			else:	 
				stroke = contour_tensor[i]
				if(len(stroke) > max_seq):
					max_seq = len(stroke)
				normalized.append(normalize_font_glyphs(stroke, font))
		normalized = pad_batch(normalized, max_seq)
		normalized_batch.append(normalized)
		counter += batch_size

#keras, the only thing of value inside tensorflow apparently, also accomplishes this (albeit unbatched):
padded_contours = tf.keras.preprocessing.sequence.pad_sequences(contour_tensor,padding='post')
padded_normalized = tf.keras.constraints.UnitNorm(axis=1)(tf.cast(padded_contours, tf.float32))
#A 'lazy' generator that loads data into memory while being iterated
batched_inputs = (padded_normalized[i:i+batch_size] for i in range(0, len(padded_normalized),2))

masking_layer = keras.layers.Masking()
lstm_layer = keras.layers.LSTM(2, return_sequences=True, input_shape=(None, 2))
#output_layer = keras.layers.Dense(5)

model = keras.Sequential([masking_layer, lstm_layer])
model.compile(loss='mae',optimizer='sgd')
model.summary()
for batch in batched_inputs:
	padded_targets = batch
	model.fit(batch, padded_targets, batch_size = batch_size)


###################################################################################################
### Overall, use of the tf.data.Dataset input API is way more trouble than its worth for prototyping at our stage
###################################################################################################
#Create a TensorFlow Dataset in order to use special functions for data preparation 
#Use of ragged tensors is crucial - unequal stroke lengths causes non-rectangular sequence data
#Ragged tensors are NOT the way to go - the feature barely has support anywhere
#dataset = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(contour_tensor, dtype=tf.float32))
#dataset = dataset.map(lambda x: normalize_font_glyphs(x, font))	#This function freezes the computer for even a small number of glyphs.Not sure what up

#If I'm understanding correctly, the use of ragged tensors removes the need for padding sequences - unsure how this works at the level of network architecture
#batched_dataset = dataset.batch(batch_size)

#Catch-22: padded datasets can only be produced if the input is rectangular, but I need padded batching to make a rectangular batch...
#batched_dataset = dataset.padded_batch(batch_size, padded_shapes=(None,None))

#iterators for debugging
#iterator = batched_dataset.make_one_shot_iterator()
#next_element = iterator.get_next()
#keras.Input(shape=(None,None), ragged=True)
####################################################################################################

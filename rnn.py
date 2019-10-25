import tensorflow as tf
from tensorflow import keras
import numpy as np
from sqlalchemy.dialects import postgresql

from models import Font, Glyph, Contour, Stroke, Point
from query import *
from corpus import *

from gutenberg import *

#def train_joyce_texts():
#text = get_joyce_texts()
joyce_keys = get_etexts('author', 'Joyce, James') 
joyce_titles = {}
joyce_texts = {}
for key in joyce_keys:
    if key in joyce_titles:
        print("Title already found in corpus, skipping: ", joyce_titles[key])
    else:
        try:
            joyce_titles[key] = get_metadata('title', key)
            joyce_texts[key] = strip_headers(load_etext(key)).strip()
        except gutenberg._domain_model.exceptions.UnknownDownloadUriException:
            print("Non-text Gutenberg entry (e.g. audiobooks), skipping: ", get_metadata('title', key))
        else:
            print("Added text: ", joyce_titles[key])

for key in joyce_texts.keys():
    print(key)
    print(len(joyce_texts[key]))
    print(joyce_titles[key])

paym = 4217
ulysses = 4300
dubliners = 2814
text = joyce_texts[paym]

#Simple char-based RNN:
#https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/text_generation.ipynb
#Prepare the text for one-hot encoding by unique characters
#def train_char_rnn(text):
vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):      
    print(repr(''.join(idx2char[item.numpy()])))

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                    return_sequences=True,
                    stateful=True,
                    recurrent_initializer='glorot_uniform'),
                    tf.keras.layers.Dense(vocab_size)])
    return model

model = build_model(vocab_size = len(vocab), embedding_dim=embedding_dim,
                    rnn_units=rnn_units, batch_size=BATCH_SIZE)
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

checkpoint_dir = './models/training_checkpoints'
# Name of the checkpoint files
import os
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
filepath=checkpoint_prefix, save_weights_only=True)

EPOCHS=25
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
#return(history)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

#The network has to be re-shaped and re-compiled for generating in certain formats
#In this case, a smaller batch size suits our generation process
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

#def generate_text(model, start_string):
# Evaluation step (generating text using the learned model)
start_string = u"Dublin"
# Number of characters to generate
num_generate = 1000

# Converting our start string to numbers (vectorizing)
input_eval = [char2idx[s] for s in start_string]
input_eval = tf.expand_dims(input_eval, 0)

# Empty string to store our results
text_generated = []

# Low temperatures results in more predictable text.
# Higher temperatures results in more surprising text.
# Experiment to find the best setting.
temperature = 0.5

# Here batch size == 1
model.reset_states()
for i in range(num_generate):
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions, 0) # remove the batch dimension
    # using a categorical distribution to predict the word returned by the model
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
    # We pass the predicted word as the next input to the model
    # along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id], 0)
    text_generated.append(idx2char[predicted_id])

print(start_string + ''.join(text_generated))

#Our sequential data is hierarchical:
#-Sequences of fonts/languages
#-Sequences of glyphs
#-Sequences of contours
#-Sequences of strokes
#Only the latter of these is truly sequential, and the concept of 'samples'
#should exist as the set of strokes on a contour. Thus, we're modelling contours
#rather than glyphs. However, we have also annotated glyphs as a separate stroke type

#Goal of our input vector = [batch_size, sequence_length, txy], where len(txy) = 3, [stroke.type, point.x, point.y]
#Directly embedding categorical stroke-type data is just one choice for sequential process of contour data.
def embedded_stroke_type_input(glyphs):
    stroke_tensor = []
    contour_tensor = []
    contour_target_tensor = []
    types = []
    type_tensor = []
    type_target_tensor = []
    type_encoding = {"L": 1, "Q": 2, "C":3, "M":4, "G":5}
    for glyph in glyphs:
        for contour in glyph.contours:
            for stroke in contour.strokes:
                #stroke_tensor.append([type_encoding[stroke.type],stroke.point.x,stroke.point.y])
                types.append([type_encoding[stroke.type]])
                stroke_tensor.append([stroke.point.x,stroke.point.y])
            contour_tensor.append(stroke_tensor)
            targets = stroke_tensor[1:]
            targets.append([0,0])
            contour_target_tensor.append(targets)
            
            type_tensor.append(types)
            targets = types[1:]
            targets.append([0])
            type_target_tensor.append(targets)
            
            stroke_tensor = []
            types = []
    return(contour_tensor,contour_target_tensor,type_tensor,type_target_tensor)

def train_arabic_glyphs(batch_size = 10):
    arab_glyphs = get_unicode_block_glyphs('Arabic')
    gg = embedded_stroke_type_input(arab_glyphs)
    contour_tensor = gg[0]
    contour_targets = gg[1]

    stroke_tensor = gg[2]
    stroke_targets = gg[3]

    #One change we would like to make is to batch first, then pad according to the maximum length of each contour,
    #rather than all the contours across all the training data. Keras should be able to support this.
    padded_contours = tf.keras.preprocessing.sequence.pad_sequences(contour_tensor,padding='post')
    padded_contour_targets = tf.keras.preprocessing.sequence.pad_sequences(contour_targets,padding='post')

    padded_strokes = tf.keras.preprocessing.sequence.pad_sequences(stroke_tensor,padding='post')
    padded_stroke_targets = tf.keras.preprocessing.sequence.pad_sequences(stroke_targets,padding='post')

    padded_normalized = tf.keras.constraints.UnitNorm(axis=1)(tf.cast(padded_contours, tf.float32))
    padded_normalized_targets = tf.keras.constraints.UnitNorm(axis=1)(tf.cast(padded_contour_targets, tf.float32))

    padded_strokes = tf.keras.backend.cast(padded_strokes, dtype='float32')
    padded_stroke_targets = tf.keras.backend.cast(padded_stroke_targets, dtype='float32')

    #A 'lazy' generator that loads data into memory while being iterated
    batched_strokes = [padded_strokes[i:i+batch_size] for i in range(0, len(padded_strokes), batch_size)]
    batched_inputs = [padded_normalized[i:i+batch_size] for i in range(0, len(padded_normalized), batch_size)]

    #Shift input sequences by one stroke to generate target data - we backpropagate on the next symbol in the sequence to train a generator
    target_strokes = [padded_stroke_targets[i:i+batch_size] for i in range(0, len(padded_stroke_targets), batch_size)]
    target_points = [padded_normalized_targets[i:i+batch_size] for i in range(0, len(padded_normalized_targets), batch_size)]

    #Most basic architecture, a stacked LSTM with a feature vector containing sequences of stroke data.
    #It is possible to parameterize this architecture, but we need empirical results to answer the following questions:
    #How many hidden layers? How many cells? Alternatives to stacked architecture? Dropout tuning? 
    #each hidden unit in a neural network trained with dropout must learn to work with a randomly chosen sample of other units. 
    #This should make each hidden unit more robust and drive it towards creating useful features on its own without relying on other hidden units to correct its mistakes.
    '''
        [batch_size,seq,[t,x,y]] -----> Zero Mask ------> N * (LSTM -------> Dropout ------->) Dense --------> [batch_size,seq,[t,x,y]] 
    ''' 
    #Make a small network for conveniently concatenating tensors - in this case our targets. 
    #Very nice to be able to make little executable architectures like this for manipulating tensors easily and quickly
    stroke_targets = keras.layers.Input(shape=(None, 1)) 
    point_targets = keras.layers.Input(shape=(None, 2))
    target_concat = keras.layers.concatenate([stroke_targets, point_targets])
    concat_model = keras.Model(inputs=[stroke_targets, point_targets], outputs=target_concat)

    stroke_input = keras.layers.Input(shape=(None, 1)) 
    point_input = keras.layers.Input(shape=(None, 2))

    concat_layer = keras.layers.concatenate([stroke_input, point_input],axis=-1) #Concatenate on the inner feature axis, -1
    masking_layer = keras.layers.Masking(input_shape=(None, 3))(concat_layer)
    lstm_layer = keras.layers.LSTM(128, return_sequences=True)(masking_layer)
    dropout_layer = keras.layers.Dropout(0.25) (lstm_layer)
    #lstm_layer_2 = keras.layers.LSTM(128, return_sequences=True)(dropout_layer)
    #dropout_layer_2 = keras.layers.Dropout(0.25)(lstm_layer_2)
    #dense_layer_2 = keras.layers.Dense(3, activation='softmax')(dropout_layer_2)
    dense_layer = keras.layers.Dense(3, activation='softmax')(dropout_layer)    
    #sequential_model = keras.Model(inputs=[stroke_input, point_input], outputs=dense_layer_2)
    sequential_model = keras.Model(inputs=[stroke_input, point_input], outputs=dense_layer)
    sequential_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    sequential_model.summary()

    epochs = 50
    for i in range(0, len(batched_inputs)):
        batch_types = batched_strokes[i]
        batch_points = batched_inputs[i]
        batched_targets = concat_model(inputs=[target_strokes[i], target_points[i]])
        sequential_model.fit([batch_types,batch_points], batched_targets, batch_size = batch_size, epochs=epochs)
    return(sequential_model)

#This should be evaluated on separate validation/test sets during a real modeling workflow.
#model.predict(x=validation_inputs[1])      Allows hyperparameter tuning by training on sequences of 'architectural parameters'/ optimizers and comparing validation results
#model.predict(x=test_inputs[1])            Allows semi-objective evaluation of network generality through an independent test set. Not exactly applicable to sequence generation problems
def predict_arabic_glyphs(sequential_model):
    arab_supplement = get_unicode_block_glyphs('Arabic Supplement')
    supplement = embedded_stroke_type_input(arab_supplement)    #Grab the datums

    supplement_contours = tf.keras.preprocessing.sequence.pad_sequences(supplement[0],padding='post')
    supplement_types = tf.keras.preprocessing.sequence.pad_sequences(supplement[2],padding='post')
    supplement_contours = tf.keras.constraints.UnitNorm(axis=1)(tf.cast(supplement_contours, tf.float32))
    supplement_types = tf.keras.backend.cast(supplement_types, dtype='float32')

    predict_supplement = concat_model([supplement_types, supplement_contours])

    #Generation routine: feed in a value, then iteratively predict based on the results of the prediction routine
    #We don't really need to make prediction data - we just start with the Glyph Generation symbol:
    #new_glyph = np.array([[[5]]],[[[0.0,0.0]]],dtype='float32')    #How deterministic is this? We might be able to perturb the x-y to get different glyphs. Play wid it
    prediction_length = 100
    preds = []
    #prediction = sequential_model.predict(new_glyph)
    prediction = sequential_model.predict([supplement_types[0:1,0:1,0:1],supplement_contours[0:1,0:1,:]])

    #Part of the problem here might be a lack of context, where we are simply predicting on the last character,
    #rather than a sequence of the previous predictions - transfer hidden state somehow?
    for i in range(0, prediction_length):
        preds.append(prediction)
        prediction = sequential_model.predict([prediction[:,0:1,0:1],prediction[:,0:1,1:]])

    #Now parse predictions into a glyph format - interpret stroke type and scale back up using font metadata (short trick might be multiply by 1000)
    #We visualize contours separately until we're sure that the RNN can properly segment glyphs with the proper stroke type, if ever
    strokes = []
    for pred in preds:
        type = pred[0][0][0] * 5
        x = pred[0][0][1] * 1000
        y = pred[0][0][2] * 1000
        strokes.append(Stroke(type='L', point=Point(x=x, y=y)))

    num_contours = 5
    contour_length = int(prediction_length / num_contours)
    predicted_contours = []
    for i in range(0, num_contours):
        new_s = strokes[i*contour_length:(i*contour_length)+contour_length]
        new_s.insert(0,Stroke(type='G', point=Point(x=0,y=0)))
        predicted_contours.append(Contour(strokes=new_s))

    predicted_glyphs = [Glyph(contours = predicted_contours[i:i+1]) for i in range(0,num_contours)]
    prediction_font = Font(name="Predictions", glyphs=predicted_glyphs)
    db.session.add(prediction_font)
    db.session.commit()

#An alternative architectural choice is encoding the stroke type separately in an embedding, and concatenating with the LSTM output 
#using the keras functional API, and training on a dense embedding of both input branches. This is likely the direction
#we will go toward. The architecture is like this:
'''
    [batch_size, seq, [x,y]] -----> Normalization -----> Concatenate [batch_size, seq, [t,x,y]] --------- Masked input ------> LSTM -----> Output [batch_size, seq, [t,x,y]]
                                                                                       /
    [batch_size, seq, t] -------> Encoding -----------> Embedding --------------------/


#This is a somewhat silly use of embeddings - we only have five dimensions.
embedding_layer = keras.layers.Embedding(input_dim=6, output_dim=1, input_length=None)(stroke_input)
milti_concat_layer = keras.layers.concatenate([embedding_layer, point_input])
multi__input_masking_layer = keras.layers.Masking(input_shape=(None, 3))(multi_concat_layer)
multi_input_lstm_layer = keras.layers.LSTM(32, return_sequences=True)(multi__input_masking_layer)
multi_input_dense_layer = keras.layers.Dense(3, activation='softmax')(multi_input_lstm_layer)
multi_input_model = keras.Model(inputs=[stroke_input, point_input], outputs=multi_input_dense_layer)
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
for i in range(0, len(batched_inputs)):
    batch = batched_inputs[i]
    batched_targets = targets_points[i]
    model.fit([batched_strokes[i], batch], [target_strokes[i], batched_targets], batch_size = batch_size)
'''

'''
Additional architectures to include:
-Encoder-decoder
-Bidirectional
'''

###################################################################################################
### Overall, use of the tf.data.Dataset input API is way more trouble than its worth for prototyping at our stage
###################################################################################################
#Used with Dataset.map() to normalize according to metadata on the source font.

'''def normalize_font_glyphs(dataset, font):
    points = []
    for point in dataset:
        x = (point[0] - font.xMin) / (font.xMax - font.xMin)
        y = (point[1] - font.yMin) / (font.yMax - font.yMin)
        points.append([x,y])
    return points'''

#Create a TensorFlow Dataset in order to use special functions for data preparation 
#Use of ragged tensors is crucial - unequal stroke lengths causes non-rectangular sequence data
#Ragged tensors are NOT the way to go - the feature barely has support anywhere
#dataset = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(contour_tensor, dtype=tf.float32))
#dataset = dataset.map(lambda x: normalize_font_glyphs(x, font))    #This function freezes the computer for even a small number of glyphs.Not sure what up

#If I'm understanding correctly, the use of ragged tensors removes the need for padding sequences - unsure how this works at the level of network architecture
#batched_dataset = dataset.batch(batch_size)

#Catch-22: padded datasets can only be produced if the input is rectangular, but I need padded batching to make a rectangular batch...
#batched_dataset = dataset.padded_batch(batch_size, padded_shapes=(None,None))

#iterators for debugging
#iterator = batched_dataset.make_one_shot_iterator()
#next_element = iterator.get_next()
#keras.Input(shape=(None,None), ragged=True)
####################################################################################################

def pad_batch(normalized, seq_length):
    for stroke in normalized:
        diff = seq_length - len(stroke)
        stroke.extend([[0,0]] * diff)
    return normalized

#Batch and normalize the data, while computing the maximum sequence length per-batch and padding accordingly
#Can't use this because keras doesn't support variable-length sequences in this way?
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
    return(normalized_batch)
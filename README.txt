#Flask app backend for our personal website and associated data science side projects. Starting with glyph generator 

Flask sqlalchemy tutorial: 
https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world
Another useful management resource for SQLite:
https://www.youtube.com/watch?v=CxCK1DkikgA

Plans for this tool: 	
	-Glyph processing
	-Text processing
	-A novel network architecture for linking strokes with text generation to give our 3D characters voices.

INSTALLATION:
requires python3, pip3, etc.

sudo apt-get install python3 venv
python3 -m venv venv
source ./venv/bin/activate
pip3 install flask
pip3 install flask_cors
pip3 install flask_sqlalchemy
pip3 install flask_migrate
pip3 install ndjson
pip3 install fontTools
pip3 install requests
pip3 install --upgrade tensorflow


FIRST-TIME STARTUP OR REBOOT:
#Delete migrations folder
#Delete old SQLite .db file
flask db init
flask db migrate
flask db upgrade
python3 initialize.py

General workflow:
-Change database representations -> 
perform a migration: flask db migrate (sometimes push code to production server)  then: flask db upgrade
"To sync the database in another system just refresh the migrations folder from source control and run the upgrade command." 
- adds the migration script to production environs

Undo migration (migrating twice to go backwards is probably not ideal)
after upgrade: flask db downgrade 
then delete the obsolete migration script in '/versions', revise your code, and migrate as usual. 
Obviously, to revert changes, revert your code

#Start app
source ./venv/bin/activate
python3 server.py
flask run

#Try this to get an interpreter for debugging
source ./venv/bin/activate
python3 -i server.py

sudo python3 train_model.py \
    --training_data=data/training.tfrecord-00000-of-00010 \
    --eval_data=data/eval.tfrecord-00000-of-00010 \
    --classes_file=data/training.tfrecord.classes \
    --model_dir=/model

#Getting data - substitute object names or use '*' to get all
gsutil -m cp "gs://quickdraw_dataset/full/simplified/cactus.ndjson" . 

TODO: 
	-Start actually generating reasonable looking glyphs
		-One good approach would be attempting to overfit a small dataset just as a proof of concept
			-Beyond that, outstanding questions:
				-Is the scale and structure of our input data sufficient (more fonts, more glyphs, different representation)?
				-Is the architecture effective for solving our problem (number of cells, number of layers)?
				-Is our training procedure well-motivated (loss function, optimizers, hyperparameters)?

	-Run the sketch-RNN model on a set of Arabic fonts to generate a demo model
		-Use the normalization scheme 'Simplified Drawings' detailed here: https://github.com/googlecreativelab/quickdraw-dataset

	-Add a way to detect when a font/dataset has already been loaded into the databse to prevent replicating datapoints when constructing alphabets, etc.
		-Completed using standard font metadata

	-Think carefully about metadata on fonts - after tens of thousands of fonts, it will be very difficult to separate differents styles of font within and across languages. Unicode blocks themselves will be very heterogeneous. Need a way to learn and query particular styles when training and generating.

	-Consider using image-based learning rather than sequence-based contour data. This would open up the possibility of using CNNs, based on:
	https://pillow.readthedocs.io/en/stable/reference/ImageFont.html
	https://github.com/erikbern/deep-fonts

	-Scale-up your dataset by scraping fonts from online aggregators:
	Sources: 
	https://www.fontspace.com/unicode/block	(Organized by unicode blocks)
	-Look into deploying and running these models on CyVerse/Azure 
	-Consider switching away from SQLite (to Postgres) for ALTER support 

	-Begin thinking about how to add large amounts of text (Qur'an, bible, novels) to the database
		-Need ways to clean up punctuation, whitespacing, headings, etc when parsing .txt/.pdfs
			-Potential sources:
				https://github.com/DH-Box/corpus-downloader
				http://corpus-db.org/

DONE:
-Find an input model for incorporating other glyph/contour data (e.g. stroke types L,Q,C,M) rather than just x-y coordinates/stroke order
	SOLUTION: Added encoded stroke types to the feature vector, which compiles/runs but unable to produce reasonable output

-Achieve the same (Glyph/Contour/Stroke/Point) format with drawings and visualize them on the front-end. 
		Major difference: Drawings do not have closed curves, but are simply lines. Thus, they may not be appropriate for the font engines we'll be using ultimately. 
			-This is the same problem as we (poorly) solved with our original glyph generator: making closed curves from atomic lines. We should focus on fonts for now
				SOLUTION: Focus on fonts and forget about front-end generation of glyphs for the time being.

-Query unicode glyphs, and loop through their linked glyph records (composites), applying offsets to achieve a sequence of x-y coordinates and strokes data. 
		-Right now, we look to only have simple, Unicode glyphs properly stored. We need to convert offset information into x-y data at some point.
			-Offsets basically define a 'moveto' symbol M in the glyph's contours, directing where a composite glyph should draw its components. 
				-In this case, the offset points to a glyph, whose contours must be included. We need a way to use offsets as queries for a glyph (and its contours) when pulling composites
					for training data and visualization purposes. The output should simply be seamless integration of contours onto the final object.
					SOLUTION: Many-to-many relationship on glyphs and contours was the right way to do composites. 

-Write a function to receive a glyph from the front-end and save it using a retrievable I.D. 

-Add font file metadata to the database - a way to detect duplicate font versions?

-Keep an eye on the font spec: https://docs.microsoft.com/en-us/typography/opentype/spec/ttch01
	-May need a way to order points in a contour to create closed curves reliably
	-Ideally, the RNN should learn that contours always end where they start to close curves

-Fuck efficient support
	-Save every single glyph inside the font, but mark non-unicode glyphs with a code of -1. Link composites explicitly with other glyph records.
-Add efficient support for glyph composites by concatenating glyph records?
	-Concatenating contour data across glyph sets (many-to-many relationship)
		-Also separate starting point data from contours because composite glyphs also contain offset information for constituent contours
			-Add simple/complex flag to glyph
			-Add contour offset array
				-List of contour offsets (for each glyph) must be a separate table, called ContourOffsets. Each offset has a foreign key to a glyph, and to a contour. We will query this table when preparing training data before running a model
					-All curves are quadratic beziers, and midpoints between control points in successive curves are implied, see Notes for more resources on TrueType parsing
						-Make it so that Points can belong to multiple Strokes to prevent storing the same points over and over
						-Alternatively, design strokes so that the end-point is implicit when constructing training data, etc.
							-We chose the latter

-Create a protocol for sending font/glyph data to the front-end for visualization purposes. 
		-Need this to verify that we are parsing glyphs properly. Visualizer should accord with normalization scheme for glyphs
			-Packaging raw database models into JSON and successfully sending to front-end, but we need to lay out the final architecture to have a common set of representations
				-SOLUTION: adding columns from the database models as needed using our custom encoder

-Find representations for each data type at each level (glyphs, strokes, prose text)
	-Two data types: drawings and glyphs from fonts are the same object, containing stroke information and xy coordinates. 
	-Drawings: connect to ontology e.g. 'cactus', 'house'
	-Glyphs: connect to language system e.g. "Hangul", "Arabic"
		-Unicode blocks, but perform the classification from external API
			-Write methods to query Alphabet API and return language character-set mappings.
				-SOLUTION: Just download the Unicode block mappings from the consortium and not worry about external crap

-Write methods for parsing Quick, Draw! data into your representations

-Write a script for initializing the DB
	-Permanently save the codeblocks, and make it possible to link glyphs to codeblocks
	-Add parsing font file to the DB intialization procedure

-Find out how to initialize the complex hierarchical relationships behind your font representations. Right now, no way to properly index foreign keys, etc.
	-SOLUTION: extensive bidirectional relationships for rich representations and personal-use database

-Learn TensorFlow!
	-Begin thinking about how to query the database to prepare training data (i.e. normalization, standardization, offsetting, etc)	
	SOLUTION: Databse to model pipeline (normalization, batching) set up. Learning Tensorflow 2.0 revamped API, and depending heavily on Keras. Basic LSTM architecture done, 
		need to consider implementing other types of architecture to achieve meaningful output for contours AND for glyphs.

-Incorporate the concept of contour direction (clock-wise or counter clock-wise) and determine it during parsing
	SOLUTION: not necessary for rendering or training glyphs so may remove in the future - possible various types of stroke metadata could augment AI training

NOTES:
Deep learning notes:
	Char rnn notes:
		-Epochs 10 -> 25, loss function continues to decrease but does not flatten. Unclear howto interpret the absolute loss function value. Is a loss < 1 considered 'convergent'?. 10 -> 25 resulted in a decrease in loss around 0.75. Anecdotally, there seem to be small improvements in the quality of the generated text, with still-obvious mistakes.

You are not alone!! https://research.gold.ac.uk/19352/1/rnn_ensemble_nips.pdf
"We also observe some interesting behaviour. When multiple models are active with roughly equal mixture weights, and the system is fed a sequence containing words or phrases that are common to all models, the probabilities for the common characters accumulate whilst probabilities specific to
individual models are suppressed, i.e. when multiple models are active the system tends towards common words and phrases."
Our solution: more rigidly link the generating model to the state of a virtual character.

Ultimate goal is sequence generation on the basis of symbols generated from gameplay data (character state machines):
https://arxiv.org/pdf/1211.3711.pdf
Generating language and glyphs from gameplay requires pairing a corpus of gameplay data with ensembles of models to create a hierarchical or state-based generator to map onto characters.

Important resources regarding true-type, open-type fonts:
https://stackoverflow.com/questions/20733790/truetype-fonts-glyph-are-made-of-quadratic-bezier-why-do-more-than-one-consecu
https://developer.apple.com/fonts/TrueType-Reference-Manual/RM01/Chap1.html#necessary

https://docs.microsoft.com/en-us/typography/opentype/spec/ttch01

We design models based on querying subsets of symbols and combining their features, essentially. They're exported as fonts, and moved to the front-end for manual editing. Front-end glyph-sets are parsed exactly like fonts, assigned a custom language name. Furthermore, custom glyphs must inhabit one unassigned area of Unicode space to accomodate our notion of a custom generated language.

-We envision this as a relatively simple RNN, with another layer to process the raw xy coordinates. CNN probably too much mustard (data too sparse for convolution)
	-Not all font glyphs have ordered strokes, likely. Not necessarily important, but might have to design the model to account for that

-By re-writing areas of Unicode space, we can eventually combine text processing with glyph generation to annotate generated text with the appropriate generated symbol sets for quite interesting effects. Finally, our ultimate intention is to add this generative routine to a Unity game, and parameterize it with player interaction and character bodies.

Ultimately the purpose of a web framework can be a flexible interactive tool for creating false languages. The basis of this flexibility is the asynchronous nature of the front and back-end, where we can continuously iterate modeling our training data, which is itself continuously iterated on the front-end. These two workflows can be performed almost simultaneously

DATA SOURCES:
https://www.myfonts.com/search//free/
https://www.google.com/get/noto/
https://github.com/adobe-fonts
https://www.fontspace.com/category/unicode
https://drive.google.com/file/d/0B0GtwTQ6IF9AU3NOdzFzUWZ0aDQ/view
#Flask app backend for our personal website and associated data science side projects. Starting with glyph generator 

Flask sqlalchemy tutorial: https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world
Another useful management resource for SQLite:
https://www.youtube.com/watch?v=CxCK1DkikgA

Plans for this tool: 	
			-Glyph processing
			-Text processing
			-A novel network architecture for linking strokes with text generation to give our 3D characters voices.

General workflow:
-Change database representations -> 
perform a migration: flask db migrate (sometimes push code to production server)  then: flask db upgrade
"To sync the database in another system just refresh the migrations folder from source control and run the upgrade command." - adds the migration script to production environs

Undo migration (migrating twice to go backwards is probably not ideal)
after upgrade: flask db downgrade 
then delete the obsolete migration script in '/versions', revise your code, and migrate as usual. Obviously, to revert changes, revert your code

#Start app
python3 server.py
flask run

sudo python3 train_model.py \
    --training_data=data/training.tfrecord-00000-of-00010 \
    --eval_data=data/eval.tfrecord-00000-of-00010 \
    --classes_file=data/training.tfrecord.classes \
    --model_dir=/model

#Getting data - substitute object names or use '*' to get all
gsutil -m cp "gs://quickdraw_dataset/full/simplified/cactus.ndjson" . 

TODO: 
	-Learn TensorFlow!
	
	-Create a protocol for sending font/glyph data to the front-end for visualization purposes. 
		-Need this to verify that we are parsing glyphs properly. Visualizer should accord with normalization scheme for glyphs

	-Query unicode glyphs, and loop through their linked glyph records (composites), applying offsets to achieve a sequence of x-y coordinates and strokes data. Achieve the same format with drawings, and make an output format that can be quickly and easily sent to the front-end for visualization
		-Run the sketch-RNN model on a set of Arabic fonts to generate a demo model
			-Begin thinking about how to query the database to prepare training data (i.e. normalization, standardization, offsetting, etc)
				-Use the normalization scheme 'Simplified Drawings' detailed here: https://github.com/googlecreativelab/quickdraw-dataset
				-Don't store raw/un-normalized font data.
			
	-Teach the model how to handle 'off-curve' points by modeling sequences of Bezier types. 

	-Begin thinking about how to add large amounts of text (Qur'an, bible, novels) to the database
		-Need ways to clean up punctuation, whitespacing, headings, etc when parsing .txt/.pdfs
			-Potential sources:
				https://github.com/DH-Box/corpus-downloader
				http://corpus-db.org/

	-Add a way to detect when a font/dataset has already been loaded into the databse to prevent replicating datapoints

	-Consider switching away from SQLite (to Postgres) for ALTER support 

DONE:
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

NOTES:

Important resources regarding true-type, open-type fonts:
https://stackoverflow.com/questions/20733790/truetype-fonts-glyph-are-made-of-quadratic-bezier-why-do-more-than-one-consecu
https://developer.apple.com/fonts/TrueType-Reference-Manual/RM01/Chap1.html#necessary

https://docs.microsoft.com/en-us/typography/opentype/spec/ttch01

We design models based on querying subsets of symbols and combining their features, essentially. They're exported as fonts, and moved to the front-end for manual editing. Front-end glyph-sets are parsed exactly like fonts, assigned a custom language name. Furthermore, custom glyphs must inhabit one unassigned area of Unicode space to accomodate our notion of a custom generated language.

-We envision this as a relatively simple RNN, with another layer to process the raw xy coordinates. CNN probably too much mustard (data too sparse for convolution)
	-Not all font glyphs have ordered strokes, likely. Not necessarily important, but might have to design the model to account for that

-By re-writing areas of Unicode space, we can eventually combine text processing with glyph generation to annotate generated text with the appropriate generated symbol sets for quite interesting effects. Finally, our ultimate intention is to add this generative routine to a Unity game, and parameterize it with player interaction and character bodies.

Ultimately the purpose of a web framework can be a flexible interactive tool for creating false languages. The basis of this flexibility is the asynchronous nature of the front and back-end, where we can continuously iterate modeling our training data, which is itself continuously iterated on the front-end. These two workflows can be performed almost simultaneously
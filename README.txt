#Flask app backend for our personal website and associated data science side projects. Starting with glyph generator 

Built largely drawing from: https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world

#As it stands, we have to transfer all external data into an internal representation for:
-Storage
-Processing (model creation)
-Rendering
Ideally these should all be the same, but currently they're all different, and have different constraints.

Rendering is likely a subset of storage representations, with ancillary parameters (color, geometry) that are never stored. We don't know enough about modeling to say how Processing representations look, but it just requries a bit of careful thinking

Plans for this tool: 	
			-Glyph processing
			-Text processing
			-A novel network architecture for linking strokes with text generation to give our 3D characters voices.

General workflow:
-Change database representations -> 
perform a migration: flask db migrate (sometimes push code to production server)  then: flask db upgrade

Undo migration (migrating twice to go backwards is probably not ideal)
after upgrade: flask db downgrade 
then delete the obsolete migration script in '/versions', revise your code, and migrate as usual. Obviously, to revert changes, revert your code


#Useful commands
'server.py' is the entry point.
DON'T FORGET export FLASK_APP=server.py !!!
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
	-Find representations for each data type at each level (glyphs, strokes, prose text)

		-Parse input JSON data for glyphs and store model in databse

	-Write methods for parsing Quick, Draw! data into your representations
	-Write method to transform database representation into React format
		-Decide on a backend/transfer representation. Would prefer JSON, but other libraries for font manipulation with Python exist:
		https://github.com/fonttools/fonttools
		https://github.com/davelab6/pyfontaine - validating input fonts


DONE:
--Write a function to receive a glyph from the front-end and save it using a retrievable I.D. 

Ultimately the purpose of a web framework can be a flexible interactive tool for creating false languages. The basis of this flexibility is the asynchronous nature of the front and back-end, where we can continuously iterate modeling our training data, which is itself continuously iterated on the front-end. These two workflows can be performed almost simultaneously
#One starts to realize that proliferation of data representations is a major source of headache

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

Ultimately we will combine these characters with tree scans/developmental models to make the portraits we considered before. 
Only some portraits should be expanded into full characters, but portraits could be displayed like artworks at some point, simple digestible things.

#Useful commands
sudo python3 train_model.py \
    --training_data=data/training.tfrecord-00000-of-00010 \
    --eval_data=data/eval.tfrecord-00000-of-00010 \
    --classes_file=data/training.tfrecord.classes \
    --model_dir=/model

#Getting data - substitute object names or use '*' to get all
gsutil -m cp "gs://quickdraw_dataset/full/simplified/cactus.ndjson" .


DON'T FORGET export FLASK_APP=server.py !!!

TODO: 
	-Find representations for each data type at each level:
		-glyphs
		-prose text
	
	-Still need to study databases to move forward
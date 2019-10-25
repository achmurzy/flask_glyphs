from server import db
from models import UnicodeBlock

#This script is for running a complete reset and rebuild of the databse 
#Delete migrations folder
#Delete old SQLite .db file
#flask db init
#flask db migrate
#flask db upgrade
#python3 initialize.py
def _initBlocks(text):
  global _blocks
  _blocks = []
  import re
  pattern = re.compile(r'([0-9A-F]+)\.\.([0-9A-F]+);\ (\S.*\S)')
  for line in text.splitlines():
    m = pattern.match(line)
    if m:
      start, end, name = m.groups()
      _blocks.append((int(start, 16), int(end, 16), name))  #This is converting from the hexadecimal stored in the raw .txt file

blocks = open("unicode_blocks.txt")
_initBlocks(blocks.read())
blocks.close()

db.create_all()	#If initializing the database the first time

for start, end, name in _blocks:
	unicode_block = UnicodeBlock(name=name, start=start, end=end)
	db.session.add(unicode_block)

#Add routines for parsing Quick, Draw by URL

#Add routines for parsing the font files

db.session.commit()

from gutenberg.query import get_etexts
from gutenberg.query import get_metadata

from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers

#Making a separate DB for text may be most responsible in light of all this
#Project Gutenberg cache - query thousands of texts very fast
from gutenberg.acquire import set_metadata_cache, get_metadata_cache
from gutenberg.acquire.metadata import SqliteMetadataCache

#Some crazy bastards annotated the entire Gutenberg dataset in order to make this
#metadata available. That's what this downloads - 3.8GB of posterity ~/gutenberg_data
#cache = SqliteMetadataCache('app.db')
#cache = get_metadata_cache()
#cache.populate()
#set_metadata_cache(cache)

#We use this metadata to add an extremely small subset of Gutenberg to our personal database

def get_joyce_texts():
	joyce_keys = get_etexts('author', 'Joyce, James') 
	joyce_titles = []
	joyce_texts = {}
	for key in joyce_keys:
		joyce_titles.append(get_metadata('title', key))
		joyce_texts[key] = strip_headers(load_etext(key)).strip()
	return(joyce_texts)
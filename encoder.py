from sqlalchemy.ext.declarative import DeclarativeMeta
import json

from models import Glyph, Contour, Stroke, Point

#Generalized JSON Encoder adapted from:
#https://stackoverflow.com/questions/5022066/how-to-serialize-sqlalchemy-result-to-json
#Usage:	json.dumps(some_glyph_obj, cls=recursive_alchemy_encoder(False, ['fields_to_expand']), check_circular=FALSE)
#Recursively expands and encodes lists on our objects, in this case the 'fields_to_expand' should be:
#['contours', 'strokes', 'points'] 
#THIS IS SUPER SLOW WHYYYYYYYYYYYYYYYYYYYYYYY
#Take a list of Glyph records queried from the database as input
def recursive_alchemy_encoder(revisit_self = False, fields_to_expand = []):
    _visited_objs = []

    class AlchemyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj.__class__, DeclarativeMeta):
                # don't re-visit self
                if revisit_self:
                    if obj in _visited_objs:
                        return None
                    _visited_objs.append(obj)

                # Allow recursion to re-visit objects shared between glyphs
                if isinstance(obj, Glyph):
                    _visited_objs.clear()
                # go through each field in this SQLalchemy class - exclude specific fields giving us problems
                fields = {}
                for field in [x for x in dir(obj) if not x.startswith('_') and x != 'metadata' and x != 'query' and x != 'query_class']:
                    val = obj.__getattribute__(field)
                    # is this field another SQLalchemy object, or a list of SQLalchemy objects?
                    if field in fields_to_expand:
                        fields[field] = val                    
                    #if isinstance(val.__class__, DeclarativeMeta) or (isinstance(val, list) and len(val) > 0 and isinstance(val[0].__class__, DeclarativeMeta)):
                        # unless we're expanding this field, stop here
                        
                # a json-encodable dict
                return fields
            return json.JSONEncoder.default(self, obj)  #This is the recursive step
    return AlchemyEncoder
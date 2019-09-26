import json
import ndjson
import math
from importlib import reload

from fontTools.ttLib import TTFont

#import uuid
#draw_id = uuid.uuid4()

from server import db
from models import Point, Stroke, Glyph, Offset, Contour, UnicodeBlock, Font, Drawing

#Takes the path to an ndjson file
#Loops through drawings and stores them
def store_drawings(path):
    draw_obj = get_ndjson(path)
    for drawing in draw_obj[1:5]:
        pic = Drawing(id=drawing['key_id'], word=drawing['word'], 
country_code=drawing['countrycode'], timestamp=drawing['timestamp'],
recognized=drawing['recognized'], contours=parse_drawing(drawing))
        db.session.add(pic)

#Extracts stroke data from a single drawing
def parse_drawing(drawing):
    contours = []
    for index, line in enumerate(drawing['drawing']):
        length = len(line[0])
        lines = []
        for i in range(length-1):
            points = [Point(x=line[0][i], y=line[1][i]), Point(x=line[0][i+1], y=line[1][i+1])]
            stroke = Stroke(type='L', points = points)
            lines.append(stroke)
        contours.append(Contour(strokes=lines))
    return contours

def block(cp):
    for start, end, name in _blocks:
        if start <= cp <= end:
            return name

def char_block(ch):
    assert len(ch) == 1, repr(ch)
    cp = ord(ch)
    for start, end, name in _blocks:
        if start <= cp <= end:
            return name

def get_ndjson(file):
    with open(file) as f:
        data = ndjson.load(f)
        return data

#Takes a path to a font file and opens it, sends it to the Alphabet
#font indexing API, and extracts relevant glyphs, with language and Unicode
#metadata. Return value is JSON, with a list of missing and present symbols
#for each language indexed by UNICODE index 
#(should tell us where to look to extract glyph data for a given language)
def parse_font(font):
    path = 'datasets/fonts/'+font
    tools_font = TTFont(path)
    font_record = Font(name=font, family=tools_font['name'].names[1].string.decode("utf-8"),
                        style=tools_font['name'].names[2].string.decode("utf-8"), 
                        ascent=tools_font['hhea'].ascent, 
                        descent=tools_font['hhea'].descent,
                        units_per_em=tools_font['head'].unitsPerEm,
                        xMin=tools_font['head'].xMin, yMin=tools_font['head'].yMin,
                        xMax=tools_font['head'].xMax, yMax=tools_font['head'].yMax)
    
    codepoints = []
    #Extract valid unicode codepoints, and constituents of composite glyphs
    for cmap in tools_font['cmap'].tables:
        if cmap.isUnicode():  
            codepoints = codepoints + list(cmap.cmap.keys())

    codepoints = list(set(codepoints))  #Ensure glyphs only parsed once

    unicode_names = []
    name_dict = {}
    #Extract the glyph data using the character code
    for code in codepoints:
        for cmap in tools_font['cmap'].tables:
            if cmap.isUnicode():    
                if code in cmap.cmap.keys():
                    unicode_names.append(cmap.cmap[code])
                    name_dict[cmap.cmap[code]] = code

    print("Candidate unicode glyphs extracted by cmap naming: " + str(len(codepoints)))
    print("Total number of glyphs in font: " + str(len(tools_font['glyf'])))
    
    #Build a glyph dictionary of contour objects for simple glyphs, or a list of
    #constituent glyph names for composite glyphs

    #Importantly, our code assumes that composite glyphs have a one-deep hierarchy
    #i.e. only composed of simple glyphs. This appears to be standard, and a part
    #of validation workflows when fonts are being created by hand
    glyphs = {}
    for name in tools_font['glyf'].glyphOrder:
        glyph = tools_font['glyf'][name]
        horizontal_metrics = tools_font['hmtx'][name]
        unicode_g = -1 
        unicode_block = None
        offsets = []
        contours = []
        if name in unicode_names:
            unicode_g = name_dict[name]
            unicode_block = db.session.query(UnicodeBlock).filter((UnicodeBlock.start <= unicode_g) & (unicode_g <= UnicodeBlock.end)).first()
        if glyph.numberOfContours < 1:  #Check for composites
            if glyph.numberOfContours < 0:
                for x in glyph.components:
                    offsets.append(Offset(x=x.x, y=x.y, glyph_name=x.glyphName)) #Only store the name - the composite_glyph reference will be back-populated, and the component glyph will be referenced later
                glyphs[name] = Glyph(unicode=unicode_g, name=name,
                                    advance_width=horizontal_metrics[0],
                                    left_side_bearing=horizontal_metrics[1],
                                    xMin=glyph.xMin, xMax=glyph.xMax, 
                                    yMin=glyph.yMin, yMax=glyph.yMax, 
                                    simple=False, contours=contours,
                                    offsets = offsets, 
                                    block=unicode_block, font=font_record)
        else:             #Extract all simple glyphs in the font
            startPt = 0
            endPt = -1
            points = [Point(x=glyph.coordinates[1][0], y=glyph.coordinates[1][1])]
            for c_ind in range(0, glyph.numberOfContours):
                startPt = endPt+1
                endPt = glyph.endPtsOfContours[c_ind]
                s_ind = 0
                strokes = []
                points = []
                flag_val = 1

                #Determine curve orientation: https://en.wikipedia.org/wiki/Curve_orientation
                clockwise = True
                hull_index = 0
                hull_y = 0
                hull_x = 0

                points.append(Point(x=glyph.coordinates[startPt][0], y=glyph.coordinates[startPt][1]))
                for xy in range(startPt+1, endPt+1):
                    new_p = None
                    if flag_val == 1:
                        if glyph.flags[xy] == 1:    #Built a line 'L'
                            strokes.append(Stroke(type='L', order=s_ind, points=points))
                            s_ind += 1
                            new_p = Point(x=glyph.coordinates[xy][0], y=glyph.coordinates[xy][1])
                            points = [new_p]
                        else:
                            new_p = Point(x=glyph.coordinates[xy][0], y=glyph.coordinates[xy][1])
                            points.append(new_p)
                    else:
                        if glyph.flags[xy] == 1:    #Built a quad 'Q'
                            strokes.append(Stroke(type='Q', order=s_ind, points=points))
                            s_ind += 1
                            new_p = Point(x=glyph.coordinates[xy][0], y=glyph.coordinates[xy][1])
                            points = [new_p] 
                        else:                       #Infer quadratics from curves
                            strokes.append(Stroke(type='Q', order=s_ind, points=points))
                            s_ind += 1
                            p1 = [points[-1].x, points[-1].y]
                            p2 = [glyph.coordinates[xy][0], glyph.coordinates[xy][1]]
                            midpoint = Point(x=(p1[0] + p2[0])/2, y=(p1[1] + p2[1])/2)
                            new_p = Point(x=glyph.coordinates[xy][0], y=glyph.coordinates[xy][1])
                            points = [midpoint, new_p]
                    flag_val = glyph.flags[xy]

                    hull_point = (hull_y == new_p.y and hull_x < new_p.x) or (hull_y < new_p.y)
                    if hull_point:    
                        hull_index = s_ind-1
                        hull_y = new_p.y
                        hull_x = new_p.x
                #We'll just have to remember to add a return to start stroke when making training data
                #points = [Point(x=strokes[-1].points[1].x, y=strokes[-1].points[1].y),
                #            Point(x=strokes[1].points[1].x, y=strokes[1].points[1].x)]
                strokes.append(Stroke(type='L', order=s_ind, points=points))

                a = strokes[hull_index - 1].points[-1]
                b = strokes[hull_index].points[-1]
                c = strokes[hull_index + 1].points[-1]
                determinant = ((b.x - a.x)*(c.y - c.y)) - ((c.x - a.x)*(b.y - a.y))
                clockwise = determinant < 0
                s_ind += 1
                strokes.append(Stroke(type='M', order=s_ind))                   #Purpose is to demarcate contours for training???
                contours.append(Contour(orientation=clockwise, strokes=strokes))
            glyphs[name] = Glyph(unicode=unicode_g, name=name,
                            advance_width=horizontal_metrics[0],
                            left_side_bearing=horizontal_metrics[1],
                            xMin=glyph.xMin, xMax=glyph.xMax, 
                            yMin=glyph.yMin, yMax=glyph.yMax, 
                            simple=True, contours=contours,
                            #offsets = offsets, 
                            block=unicode_block, font=font_record) 
    
    #This adds absolutely everything because font is linked to everything else
    db.session.add(font_record)

    #MAKE ASSOCIATION TABLE AND MAKE GLYPHS/CONTOURS MANY-TO-MANY
    #Link composite glyphs to their constituent simple forms
    composite_glyphs = db.session.query(Glyph).filter(Glyph.simple == False).all()
    for composite in composite_glyphs:
        contours = []
        for component in composite.offsets:
            component_glyph = db.session.query(Glyph).filter(Glyph.name == component.glyph_name).first()
            new_contour = Contour(strokes=[Stroke(type='M', order=0, points=[Point(x=component.x, y=component.y)])], glyphs=[composite])
            db.session.add(new_contour)
            contours.append(new_contour)
            contours.extend(component_glyph.contours)

        composite.contours = contours
            
    db.session.commit()
    return glyphs

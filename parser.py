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
            point = [Point(x=line[0][i], y=line[1][i]), Point(x=line[0][i+1], y=line[1][i+1])]
            stroke = Stroke(type='L', point = point)
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

def parse_noto_sans_display():
    from os import listdir
    fonts = listdir('datasets/fonts/NotoSansDisplay')
    for font in fonts:
        ttfont = TTFont('datasets/fonts/NotoSansDisplay/'+font)
        for name in ttfont['name'].names:     
            print(name.string.decode("utf-8"))
        parse_font('NotoSansDisplay/'+font)

def check_font_record_duplicate(new_font):
    match = Font.query.filter_by(family = new_font.family).first()
    return match != None

#Takes a path to a font file and opens it, sends it to the Alphabet
#font indexing API, and extracts relevant glyphs, with language and Unicode
#metadata. Return value is JSON, with a list of missing and present symbols
#for each language indexed by UNICODE index 
#(should tell us where to look to extract glyph data for a given language)
#TTF table specification: https://docs.microsoft.com/en-us/typography/opentype/spec/name
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
    duplicate_font = check_font_record_duplicate(font_record)
    if duplicate_font:
        print("Font family already present: " + font_record.family)
        return
    else:
        print("Parsing new font family: " + font_record.family)

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
            for c_ind in range(0, glyph.numberOfContours):
                startPt = endPt+1
                endPt = glyph.endPtsOfContours[c_ind]
                s_ind = 1
                strokes = []
                flag_val = 1

                #Determine curve orientation: https://en.wikipedia.org/wiki/Curve_orientation
                clockwise = True
                hull_index = 0
                hull_y = 0
                hull_x = 0

                #Demarcate a new contour, possibly a new glyph
                strokes.append(Stroke(type="G" if c_ind == 0 else "M", order=0, point=Point(x=glyph.coordinates[startPt][0], y=glyph.coordinates[startPt][1])))
                for xy in range(startPt+1, endPt+1):
                            #LAST POINT ON-CURVE
                    if flag_val == 1:
                        if glyph.flags[xy] == 1:    #THIS POINT ON-CURVE - Build a line 'L'
                            #strokes.append(Stroke(type='L', order=s_ind, points=points))
                            strokes.append(Stroke(type='L', order=s_ind, point=Point(x=glyph.coordinates[xy][0], y=glyph.coordinates[xy][1])))
                        else:                       #THIS POINT OFF-CURVE - Build a control point 'C'
                            strokes.append(Stroke(type='C', order=s_ind, point=Point(x=glyph.coordinates[xy][0], y=glyph.coordinates[xy][1])))
                    else:   #LAST POINT OFF-CURVE
                        if glyph.flags[xy] == 1:    #THIS POINT ON-CURVE - Build a quadratic 'Q'
                            strokes.append(Stroke(type='Q', order=s_ind, point=Point(x=glyph.coordinates[xy][0], y=glyph.coordinates[xy][1]))) 
                        else:                       #THIS POINT OFF-CURVE - Infer a quadratic 'Q' from points and build a control point 'C' 
                            #We are dealing with a series of 'off-curve' points
                            #Infer implicit 'on-curve' points as the midpoint between two control points
                            p1 = [strokes[-1].point.x, strokes[-1].point.y]
                            p2 = [glyph.coordinates[xy][0], glyph.coordinates[xy][1]]
                            midpoint = Point(x=(p1[0] + p2[0])/2, y=(p1[1] + p2[1])/2)
                            strokes.append(Stroke(type='Q', order=s_ind, point=midpoint))
                            s_ind += 1
                            strokes.append(Stroke(type='C', order=s_ind, point=Point(x=p2[0], y=p2[1])))
                    s_ind += 1
                    flag_val = glyph.flags[xy]

                    new_p = strokes[-1].point
                    hull_point = (hull_y == new_p.y and hull_x < new_p.x) or (hull_y < new_p.y)
                    if hull_point:    
                        hull_index = s_ind-1
                        hull_y = new_p.y
                        hull_x = new_p.x
                #We'll just have to remember to add a return to start stroke when making training data
                #Having a line between the last and first point is implicit behavior in all kinds of drawing applications
                #We will handle this in our front-end visualization and Unity's font engine
                #points = [Point(x=strokes[-1].points[1].x, y=strokes[-1].points[1].y),
                #            Point(x=strokes[1].points[1].x, y=strokes[1].points[1].x)]
                #strokes.append(Stroke(type='L', order=s_ind, points=points))

                num_strokes = len(strokes)
                a = strokes[hull_index- 1].point
                b = strokes[hull_index].point
                c = strokes[(hull_index + 1)%num_strokes].point
                determinant = ((b.x - a.x)*(c.y - c.y)) - ((c.x - a.x)*(b.y - a.y))
                clockwise = determinant < 0
                s_ind += 1
                #strokes.append(Stroke(type='M', order=s_ind))                   #Purpose is to demarcate contours for training???
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
    #Link composite glyphs to their constituent simpler contours
    #Only query composite glyphs from the current font using a join
    composite_glyphs = Glyph.query.filter_by(simple = False, font = font_record).all()
    for composite in composite_glyphs:
        contours = []
        for i in range(0, len(composite.offsets)):
            component = composite.offsets[i]
            component_glyph = Glyph.query.filter_by(name = component.glyph_name, font = font_record).first()
            new_contour = Contour(strokes=[Stroke(type="G" if i == 0 else "M", order=i, point=Point(x=component.x, y=component.y))], glyphs=[composite])
            db.session.add(new_contour)
            contours.append(new_contour)
            contours.extend(component_glyph.contours)

        composite.contours = contours
            
    db.session.commit()
    return glyphs

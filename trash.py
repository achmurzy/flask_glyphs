APIrootURL = 'http://alphabet-type.com/en/tools/api/'
list_url = APIrootURL  + 'v1.0/tools/charset-checker/list'

#Here we'll need to check an arbitrary font against all possible language 
#systems to determine which glyphs are viable, and store them
data = {"languagesystems" : ['en'],
        "codepoints" : codepoints}
        
response = requests.post(list_url, data=data)
if response.status_code == 200:
   support = response.json()
else:
   print response.text 

    #Standardized language codes: https://en.wikipedia.org/wiki/ISO_639-3
languages = {}
language_json = open("alphabet_support.json")
language_codes = json.loads(language_json.read())
language_json.close()
for language in language_codes:
    languages[language['code']] = language['name']

    arabtype_test = open("fonts/arabtype_languagesystems_response.json")
    support = json.loads(arabtype_test.read())
    arabtype_test.close()

    #Current issue: languages 'require' the same glyphs over and over again
    #Unicode blocks are a much more robust approach to this issue
    language_systems = support['languagesystems']
    for language in language_systems:
        #Make entries for present, required glyphs in a given language system
        present_required_codes = language_systems[language]['required']['present']
        for code in present_required_codes:
            #Extract the glyph data using the character code
            names = []
            name_dict = {}
            for cmap in tools_font['cmap'].tables:
                if code in cmap.cmap.keys():
                    names.append(cmap.cmap[code])
                    name_dict[cmap.cmap[code]] = code
            
            print("Total glyph mappings from cmap: " + str(len(names)))
            print("Total size of glyf table: " + str(len(tools_font['glyf'].glyphOrder)))
            #Back-index the unicode point and add it to the glyph objects
            font_glyphs = []
            for x in names:
                n_g = tools_font['glyf'][x]
                n_g.unicode = name_dict[x]
                n_g.name = x
                font_glyphs.append(n_g)

            print("Candidate glyphs extracted by cmap naming: " + str(len(font_glyphs)))
            glyphs = []
            for glyph in font_glyphs:
                contours = []
                if glyph.numberOfContours < 1:
                    continue
                    #print("Composite glyph or empty, skip for now")
                else:
                    #print("Parsing glyph: " + glyph.name)
                    strokes = []
                    startPt = 1
                    endPt = glyph.endPtsOfContours[0]
                    points = [Point(x=glyph.coordinates[1][0], y=glyph.coordinates[1][1])]
                    for c_ind in range(0, glyph.numberOfContours-1):
                        flag_val = 1
                        cubic_flag = 0
                        for xy in range(startPt, endPt):
                            points.append(Point(x=glyph.coordinates[xy][0], y=glyph.coordinates[xy][1]))
                            if flag_val == 1:
                                if glyph.flags[xy] == 1:    #Built a line 'L'
                                    strokes.append(Stroke(type='L', points=points))
                                    points = []
                            else:
                                if glyph.flags[xy] == 1:    #Built a quad 'Q'
                                    if cubic_flag == 1:
                                        strokes.append(Stroke(type='C', points=points))
                                        points = []
                                        cubic_flag = 0
                                    else:
                                        strokes.append(Stroke(type='Q', points=points))
                                        points = [] 
                                else:                       #Next must be cubic
                                    cubic_flag = 1
                            flag_val = glyph.flags[xy]
                        new_contour = Contour(strokes=strokes)
                        startPt = endPt
                        endPt = glyph.endPtsOfContours[c_ind]
                    glyphs.append(Glyph(unicode=glyph.unicode, xMin=glyph.xMin, 
                                    xMax=glyph.xMax, yMin=glyph.yMin, 
                                    yMax=glyph.yMax, contours=contours))

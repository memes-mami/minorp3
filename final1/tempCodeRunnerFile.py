@app.route('/basetemp2')
def slideshow():
    im = [
        'cv.png',
        'cloud.jpg',
        'tree.jpg',
        'aud.jpg',
        'ent.jpg',
        'mmm.jpg',
        'srk1.jpg'
    ]
    return render_template('basetemp2.html', images=im)
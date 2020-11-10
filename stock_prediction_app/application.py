from flask import Flask, render_template,request, jsonify
import pandas as pd
import numpy as np
import os


application = Flask(__name__, static_folder='static')



@application.route('/', methods=['GET','POST'])
def homepage(): 
    return render_template('app.html')


@application.route('/trainModel', methods=['GET','POST'])
def imageUpload():
    #Get data from the Get request...
    # Then train model for the specific stock...
    # Then return data with json....
    
    """
    content={}
    if(request.method=="POST"):
        print(request.files)
        imagefile=request.files.get('fileUpload')
        testimage = Image.open(imagefile)
        out=dogCLF.getPreds(testimage)
        breeds=dogCLF.getBreeds(out)
        #print(breeds)
        #return render_template('app.html')
        content={'pred1':breeds[0],'pred2':breeds[1],'pred3':breeds[2]}
        #content={'pred1':'dog1','pred2':'dog2','pred3':'dog3'}
    return jsonify(content)
    #return render_template('app.html',data=content)
    """




if __name__=="__main__":
    application.run()
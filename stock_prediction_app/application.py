from flask import Flask, render_template,request, jsonify
import pandas as pd
import numpy as np
import os
from model import main

application = Flask(__name__, static_folder='static')

@application.route('/', methods=['GET','POST'])
def homepage(): 
    imageUpload()
    return render_template('app.html')


@application.route('/getStock',methods=['GET','POST'])
def getStock():
    content={}
    if(request.method=="GET"):
        print(request.args)
        stockName=request.args.get('stockName')
        #Check to see if this stockname is legit
        content['stockName']=stockName
    return jsonify(content)


@application.route('/trainModel', methods=['GET','POST'])
def trainModel():
    #Get data from the Get request...
    # Then train model for the specific stock...
    # Then return data with json....
    content={}
    if(request.method=="GET"):
        print(request.args)
        stockName=request.args.get('stockName')
    return jsonify(content)
    #return render_template('app.html',data=content)




if __name__=="__main__":
    application.run()
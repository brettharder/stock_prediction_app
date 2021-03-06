from flask import Flask, render_template,request, jsonify
import pandas as pd
import numpy as np
import os
import model

application = Flask(__name__, static_folder='static')

@application.route('/', methods=['GET','POST'])
def homepage(): 
    return render_template('app.html')


@application.route('/getStock',methods=['GET','POST'])
def getStock():
    content={}
    if(request.method=="GET"):
        print(request.args)
        stockName=request.args.get('stockName')
        #Check to see if this stockname is legit
        shortStockName=model.confirmStock(stockName)
        if(pd.notnull(shortStockName)):
            content['stockName']=shortStockName
        else:
            content['stockName']=False
    return jsonify(content)


@application.route('/trainModel', methods=['GET','POST'])
def trainModel():
    #Get data from the Get request...
    # Then train model for the specific stock...
    # Then return data with json....
    content={}
    if(request.method=="GET"):
        #print(request.args)
        stockName=request.args.get('stockName')
        print(stockName)
        modelType =request.args.get('modelType')
        print(modelType)
        returnData=model.main(stockName,modelType)
        #print(returnData) 
        #content=returnData
        plotlyDiv=model.plotThis(returnData['train']['x'],returnData['train']['y'],
            returnData['test']['x'],returnData['test']['y'],
            returnData['actual']['x'],returnData['actual']['y'],
            modelType=modelType
        )       
        content['plotDiv']=plotlyDiv
    return jsonify(content)




if __name__=="__main__":
    application.run()
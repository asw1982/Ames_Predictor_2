# -*- coding: utf-8 -*-


from flask import Flask, render_template, url_for ,request , redirect
#from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os 

from smiles2ames import *

# create empty model with the hyperparameter 
nCV = 10
num_layer = 3
dropout_rate=0.2051210302624803
hidden_channels = 64 
num_node_features = 79 
num_classes = 1

list_trained_model =[]
for i in range(10):
    loaded_model = GCN(hidden_channels,num_node_features, num_classes, dropout_rate, num_layer) 
    loaded_model.load_state_dict(torch.load("model_GNN"+ str(i)+ ".pth"))
    list_trained_model.append(loaded_model)




app = Flask(__name__)

picFolder =os.path.join('static','pics')
app.config['UPLOAD_FOLDER']= picFolder
#app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///test.db'

#db = SQLAlchemy(app)


  
    
@app.route('/', methods= ['POST','GET'])

def index():
    pred_result =""
    smiles_input = ""
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'],"blokdiagram2.png")
    if request.method =='POST':
        smiles_input = request.form['content']
        
        pred_result = smiles_to_ames(smiles_input)
        return render_template('index.html',smiles_input=smiles_input, pred_result=pred_result, user_image=pic1)
        #request.form['result']=pred_result    
    else:
        return render_template('index.html',smiles_input=smiles_input,pred_result=pred_result, user_image=pic1)


    
if __name__=="__main__":
    app.run(debug=True)
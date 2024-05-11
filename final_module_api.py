from flask import Flask,request,jsonify,render_template
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,ConfusionMatrixDisplay,roc_curve,auc
from joblib import dump,load
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

#   CREATE FLASK APP
app=Flask(__name__)
CORS(app)

#   CONNECT POST API CALL
#   http://localhost:5000/
#   @-->Decorator function
@app.route('/',methods=['GET'])
def predict():
    return render_template('index.html')

class AppData:
    def __init__(self):
        self.df = pd.DataFrame()
        self.classes = []
        self.X_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_test = pd.DataFrame()

    def load_data(self,file_name_dict):
        file_name=file_name_dict['file_name']
        self.df = pd.read_csv(file_name)
        if (self.df[self.df.columns[-1]].nunique()>=10 or len(self.df.columns)!=8):
            return {'message' : 'This is a regression problem.This web application cannot solve regression problems'}
        self.classes=self.df['class'].unique()
        return {'message': 'Data has been successfully loaded from local drive',
                'class_names' : np.array(self.classes).tolist()}

    def prepare_data(self): 
        self.df=self.df.drop('Unnamed: 0',axis=1)
        X=self.df.drop('class',axis=1)
        y=self.df['class']
        X=pd.get_dummies(X,drop_first=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=101)
        return {'message': 'Data has been successfully prepared for training'}
    
    def logistic_regression(self):
        log_model = LogisticRegression(solver='saga',multi_class="ovr",max_iter=5000)
        cross_val_score(log_model,self.X_train,self.y_train,scoring='accuracy',cv=10)
        log_model.fit(self.X_train,self.y_train)
        dump(log_model,'logistic_regression.pkl')
        return {'message': 'Logistic regression has been successfully trained and saved locally'}

    def decision_tree(self):
        model = DecisionTreeClassifier()
        cross_val_score(model,self.X_train,self.y_train,scoring='accuracy',cv=10)
        model.fit(self.X_train,self.y_train)
        dump(model,'decision_tree.pkl')
        return {'message': 'Decision Tree has been successfully trained and saved locally'}

    def random_forest(self):
        model = RandomForestClassifier(n_estimators=10,random_state=101)
        cross_val_score(model,self.X_train,self.y_train,scoring='accuracy',cv=10)
        model.fit(self.X_train,self.y_train)
        dump(model,'random_forest.pkl')
        return {'message': 'Random Forest has been successfully trained and saved locally'}

    def knn(self):
        model=KNeighborsClassifier(n_neighbors=8)
        cross_val_score(model,self.X_train,self.y_train,scoring='accuracy',cv=10)
        model.fit(self.X_train,self.y_train)
        dump(model,'knn.pkl')
        return {'message': 'K Nearest Neighbors has been successfully trained and saved locally'}

    def svm(self):
        model = SVC(kernel='linear', C=1000)
        cross_val_score(model,self.X_train,self.y_train,scoring='accuracy',cv=10)
        model.fit(self.X_train,self.y_train)
        dump(model,'svm.pkl')
        return {'message': 'Support Vector Machine has been successfully trained and saved locally'}

    def cal_confusion_matrix(self,model_name):
        if(model_name['model_name']=="Logistic Regression"):
            loaded_model=load('logistic_regression.pkl')
        elif(model_name['model_name']=="Decision Tree"):
            loaded_model=load('decision_tree.pkl')
        elif(model_name['model_name']=="Random Forest"):
            loaded_model=load('random_forest.pkl')
        elif(model_name['model_name']=="K Nearest Neighbors"):
            loaded_model=load('knn.pkl')
        else:
            loaded_model=load('svm.pkl')
        y_pred=loaded_model.predict(self.X_test)
        cm=confusion_matrix(self.y_test,y_pred)
        return {'Confusion Matrix': np.array(cm).tolist()}

    def cal_sensitivity_specitivity(self,model_name,sen_spec):
        cm=self.cal_confusion_matrix(model_name)
        cm=cm["Confusion Matrix"]
        True_Positive = np.diag(cm)
        False_Positive = np.sum(cm, axis=0) - True_Positive
        False_Negative = np.sum(cm, axis=1) - True_Positive
        True_Negative = np.sum(cm) - (True_Positive + False_Positive + False_Negative)
        if (sen_spec=="Sensitivity"):
            sensitivity = True_Positive / (True_Positive + False_Negative)
            return {sen_spec: np.array(sensitivity).tolist()}
        else:
            specificity = True_Negative / (True_Negative + False_Positive)
            return {sen_spec: np.array(specificity).tolist()}

    def plot_roc(self, clf, n_classes, model_name, figsize=(5,5)):
        if(model_name['model_name']=="Logistic Regression" or model_name['model_name']=="Support Vector Machine"):
            y_score = clf.decision_function(self.X_test)
        else:
            y_score = clf.predict_proba(self.X_test)
        # structures
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # calculate dummies once
        y_test_dummies = pd.get_dummies(self.y_test, drop_first=False).values
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # roc for each class
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for ' + str(self.classes[i]))
        ax.legend(loc="best")
        ax.grid(alpha=.4)
        sns.despine()
        for i in range(1,6):
            if os.path.exists('static/images/roc' + str(i) + '.jpg'):
                os.remove('static/images/roc' + str(i) + '.jpg')
        if(model_name['model_name']=="Logistic Regression"):
            plt.savefig('static/images/roc1.jpg')
            return {'ImageURL':'/static/images/roc1.jpg'}
        elif(model_name['model_name']=="Decision Tree"):
            plt.savefig('static/images/roc2.jpg')
            return {'ImageURL':'/static/images/roc2.jpg'}
        elif(model_name['model_name']=="Random Forest"):
            plt.savefig('static/images/roc3.jpg')
            return {'ImageURL':'/static/images/roc3.jpg'}
        elif(model_name['model_name']=="K Nearest Neighbors"):
            plt.savefig('static/images/roc4.jpg')
            return {'ImageURL':'/static/images/roc4.jpg'}
        else:
            plt.savefig('static/images/roc5.jpg')
            return {'ImageURL':'/static/images/roc5.jpg'}

    def calculate_auc(self, clf, n_classes, model_name):
        if(model_name['model_name']=="Logistic Regression" or model_name['model_name']=="Support Vector Machine"):
            y_score = clf.decision_function(self.X_test)
        else:
            y_score = clf.predict_proba(self.X_test)
        # structures
        fpr = dict()
        tpr = dict()
        auc_list=[]
        # calculate dummies once
        y_test_dummies = pd.get_dummies(self.y_test, drop_first=False).values
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
            auc_list.append(auc(fpr[i], tpr[i]))
        return {'AUC' : auc_list}

    def sensitivity_at_specificity_90(self, clf, n_classes, model_name):
        if(model_name['model_name']=="Logistic Regression" or model_name['model_name']=="Support Vector Machine"):
            y_score = clf.decision_function(self.X_test)
        else:
            y_score = clf.predict_proba(self.X_test)
        y_test_dummies = pd.get_dummies(self.y_test, drop_first=False).values
        sen_at_spec_90=[]
        desired_spec = 0.90
        for i in range(n_classes):
            fpr, tpr, thresholds = roc_curve(y_test_dummies[:, i], y_score[:, i])
            idx = np.argmax(fpr >= (1 - desired_spec))
            sen_at_spec_90.append(round(tpr[idx], 4))
        return {"Sensitivity at specificity = 0.90" : sen_at_spec_90}



app_data = AppData()


@app.route('/load_data', methods=['POST'])
def load_data():
    file_name=request.json
    return jsonify(app_data.load_data(file_name))

@app.route('/prepare_data', methods=['GET'])
def prepare_data():
    return jsonify(app_data.prepare_data())

@app.route('/logistic_regression', methods=['GET'])
def logistic_regression():
    return jsonify(app_data.logistic_regression())

@app.route('/decision_tree', methods=['GET'])
def decision_tree():
    return jsonify(app_data.decision_tree())

@app.route('/random_forest', methods=['GET'])
def random_forest():
    return jsonify(app_data.random_forest())

@app.route('/knn', methods=['GET'])
def knn():
    return jsonify(app_data.knn())

@app.route('/svm', methods=['GET'])
def svm():
    return jsonify(app_data.svm())

@app.route('/cal_confusion_matrix', methods=['POST'])
def cal_confusion_matrix():
    model_name=request.json
    return jsonify(app_data.cal_confusion_matrix(model_name))

@app.route('/cal_sensitivity', methods=['POST'])
def cal_sensitivity():
    model_name=request.json
    return jsonify(app_data.cal_sensitivity_specitivity(model_name,"Sensitivity"))

@app.route('/cal_specificity', methods=['POST'])
def cal_specificity():
    model_name=request.json
    return jsonify(app_data.cal_sensitivity_specitivity(model_name,"Specificity"))

@app.route('/roc_curve', methods=['POST'])
def plot_roc():
    model_name=request.json
    if(model_name['model_name']=="Logistic Regression"):
            loaded_model=load('logistic_regression.pkl')
    elif(model_name['model_name']=="Decision Tree"):
        loaded_model=load('decision_tree.pkl')
    elif(model_name['model_name']=="Random Forest"):
        loaded_model=load('random_forest.pkl')
    elif(model_name['model_name']=="K Nearest Neighbors"):
        loaded_model=load('knn.pkl')
    else:
        loaded_model=load('svm.pkl')
    return jsonify(app_data.plot_roc(loaded_model, 4, model_name))

@app.route('/auc', methods=['POST'])
def calc_auc():
    model_name=request.json
    if(model_name['model_name']=="Logistic Regression"):
            loaded_model=load('logistic_regression.pkl')
    elif(model_name['model_name']=="Decision Tree"):
        loaded_model=load('decision_tree.pkl')
    elif(model_name['model_name']=="Random Forest"):
        loaded_model=load('random_forest.pkl')
    elif(model_name['model_name']=="K Nearest Neighbors"):
        loaded_model=load('knn.pkl')
    else:
        loaded_model=load('svm.pkl')
    return jsonify(app_data.calculate_auc(loaded_model, 4, model_name))

@app.route('/sen_at_spec_90', methods=['POST'])
def sen_at_spec_90():
    model_name=request.json
    if(model_name['model_name']=="Logistic Regression"):
            loaded_model=load('logistic_regression.pkl')
    elif(model_name['model_name']=="Decision Tree"):
        loaded_model=load('decision_tree.pkl')
    elif(model_name['model_name']=="Random Forest"):
        loaded_model=load('random_forest.pkl')
    elif(model_name['model_name']=="K Nearest Neighbors"):
        loaded_model=load('knn.pkl')
    else:
        loaded_model=load('svm.pkl')
    return jsonify(app_data.sensitivity_at_specificity_90(loaded_model, 4, model_name))

if __name__=='__main__':
    app.run(debug=True)
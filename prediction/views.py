from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.shortcuts import render
import multiprocessing
from django.http import HttpResponse, HttpResponseRedirect
import subprocess
import distutils.dir_util
from django.conf import settings
import os, datetime

from django.core.mail import send_mail
from django.core.mail import EmailMessage
#from pred import *
#from . import pred

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

import datetime
import os
from io import BytesIO
import base64
from sklearn import metrics

def index(request):
    return render(request, 'index.html', {})

def decision(request):
	if request.POST and request.FILES:
		try:
			global filename, id, emailid
			emailid = request.POST['emailid']
			filename = request.FILES['csv_file']

			id = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
			dir_path = os.path.join('F:\\Django-project\\classification\\media\\result', id)

			folder = os.mkdir(dir_path)

			dataset = pd.read_csv(filename)

			#decision_result=int(pred_decision(dataset))

			X = dataset.iloc[:,[2,3]].values
			y = dataset.iloc[:,4].values

			from sklearn.model_selection import train_test_split
			X_train1,X_test1,y_train1,y_test1=train_test_split(X,y,test_size = 0.25, random_state = 0)

			from sklearn.preprocessing import StandardScaler
			sc_X = StandardScaler()
			X_train1 = sc_X.fit_transform(X_train1)
			X_test1 = sc_X.transform(X_test1)

			from sklearn.tree import DecisionTreeClassifier
			classifier = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
			classifier.fit(X_train1,y_train1)
			y_pred1 = classifier.predict(X_test1)

			from sklearn.metrics import classification_report
			a=(classification_report(y_test1, y_pred1))

			a1="Decision tree Accuracy :",metrics.accuracy_score(y_test1, y_pred1)

			with open(os.path.join(dir_path,id + ".log"), 'w') as file:
				file.write("\n\n----- DECISION TREE -----  \n\n\n")



			with open(os.path.join(dir_path,id + ".log"), 'a') as file:
				for line in a,a1:
					file.write(str(line))
				file.write("\n DECISION TREE SUCCESSFULLY COMPLETED\n\n\n")

			from matplotlib.colors import ListedColormap
			X_set1, y_set1=X_test1,y_test1
			X11,X21 = np.meshgrid(np.arange(start=X_set1[:,0].min()-1,stop=X_set1[:,0].max()+1,step=0.01),np.arange(start=X_set1[:,1].min()-1,stop=X_set1[:,1].max()+1,step=0.01))

			plt.contourf(X11,X21, classifier.predict(np.array([X11.ravel(),X21.ravel()]).T).reshape(X11.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
			plt.xlim(X11.min(),X11.max())
			plt.ylim(X21.min(),X21.max())


			for i,j in enumerate(np.unique(y_set1)):
				plt.scatter(X_set1[y_set1==j,0],X_set1[y_set1==j,1],c=ListedColormap(('red','green'))(i),label=j)
				plt.title('Decision tree classifier(Testing set)')
				plt.xlabel('Age')
				plt.ylabel('Estimated salary')
				
				

			sample_file_name = "decision-test-result"
			plt.savefig(dir_path + "/" + sample_file_name)

			buffer=BytesIO()
			plt.savefig(buffer,format='png')
			buffer.seek(0)
			image_png1 = buffer.getvalue()
			buffer.close()
			#plt.cla()
			plt.clf()

			graphic1 = base64.b64encode(image_png1)
			graphic1 = graphic1.decode('utf-8')

			#return render(request, 'progress.html',{'graphic1':graphic1})


			from sklearn.model_selection import train_test_split
			X_train2,X_test2,y_train2,y_test2=train_test_split(X,y,test_size = 0.25, random_state = 0)

			from sklearn.preprocessing import StandardScaler
			sc_X = StandardScaler()
			X_train2 = sc_X.fit_transform(X_train2)
			X_test2 = sc_X.transform(X_test2)




			
			from sklearn.svm import SVC
			classifier = SVC(kernel = 'linear', random_state = 0)
			classifier.fit(X_train2, y_train2)

			y_pred2 = classifier.predict(X_test2)

			from sklearn.metrics import classification_report
			b=(classification_report(y_test2, y_pred2))
			b1="SVM Accuracy :",metrics.accuracy_score(y_test2, y_pred2)

			with open(os.path.join(dir_path,id + ".log"), 'a') as file:
				file.write("\n\n\n----- SVM -----\n")


			

			with open(os.path.join(dir_path,id + ".log"), 'a') as file:
				for line in b,b1:
					file.write(str(line))
				file.write("\n SVM SUCCESSFULLY COMPLETED\n\n\n")


			from matplotlib.colors import ListedColormap
			X_set2, y_set2 = X_test2, y_test2
			X12, X22 = np.meshgrid(np.arange(start = X_set2[:, 0].min() - 1, stop = X_set2[:, 0].max() + 1, step = 0.01),
                     		np.arange(start = X_set2[:, 1].min() - 1, stop = X_set2[:, 1].max() + 1, step = 0.01))
			plt.contourf(X12, X22, classifier.predict(np.array([X12.ravel(), X22.ravel()]).T).reshape(X12.shape),
            				alpha = 0.75, cmap = ListedColormap(('red', 'green')))
			plt.xlim(X12.min(), X12.max())
			plt.ylim(X22.min(), X22.max())
			for i, j in enumerate(np.unique(y_set2)):
				plt.scatter(X_set2[y_set2 == j, 0], X_set2[y_set2 == j, 1],c = ListedColormap(('red', 'green'))(i), label = j)
				plt.title('SVM (Testing set)')
				plt.xlabel('Age')
				plt.ylabel('Estimated Salary')
				


			sample_file_name = "svm-test-result"
			plt.savefig(dir_path + "/" + sample_file_name)

			buffer=BytesIO()
			plt.savefig(buffer,format='png')
			buffer.seek(0)
			image_png2 = buffer.getvalue()
			buffer.close()
			#plt.cla()
			plt.clf()

			graphic2 = base64.b64encode(image_png2)
			graphic2 = graphic2.decode('utf-8')

			#return render(request, 'progress.html',{'graphic':graphic2})

			from sklearn.model_selection import train_test_split
			X_train3,X_test3,y_train3,y_test3=train_test_split(X,y,test_size = 0.25, random_state = 0)

			from sklearn.preprocessing import StandardScaler
			sc_X = StandardScaler()
			X_train3 = sc_X.fit_transform(X_train3)
			X_test3 = sc_X.transform(X_test3)

			from sklearn.ensemble import RandomForestClassifier
			classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0 )
			classifier.fit(X_train3,y_train3)
			y_pred3 = classifier.predict(X_test3)

			from sklearn.metrics import classification_report
			c=(classification_report(y_test3, y_pred3))
			c1="RANDOM FOREST Accuracy :",metrics.accuracy_score(y_test3, y_pred3)

			with open(os.path.join(dir_path,id + ".log"), 'a') as file:
				file.write("\n\n\n----- RANDOMFOREST ----- \n")

			

			with open(os.path.join(dir_path,id + ".log"), 'a') as file:
				for line in c,c1:
					file.write(str(line))
				file.write("\n RANDOMFOREST SUCCESSFULLY COMPLETED\n")


			from matplotlib.colors import ListedColormap
			X_set3, y_set3=X_test3,y_test3
			X13,X23 = np.meshgrid(np.arange(start=X_set3[:,0].min()-1,stop=X_set3[:,0].max()+1,step=0.01),np.arange(start=X_set3[:,1].min()-1,stop=X_set3[:,1].max()+1,step=0.01))

			plt.contourf(X13,X23, classifier.predict(np.array([X13.ravel(),X23.ravel()]).T).reshape(X13.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
			plt.xlim(X13.min(),X13.max())
			plt.ylim(X23.min(),X23.max())


			for i,j in enumerate(np.unique(y_set3)):
				plt.scatter(X_set3[y_set3==j,0],X_set3[y_set3==j,1],c=ListedColormap(('red','green'))(i),label=j)
				plt.title('Random Forest classifier(Testing set)')
				plt.xlabel('Age')
				plt.ylabel('Estimated salary')
				

			sample_file_name = "randomforest-test-result"
			plt.savefig(dir_path + "/" + sample_file_name)

			buffer=BytesIO()
			plt.savefig(buffer,format='png')
			buffer.seek(0)
			image_png3 = buffer.getvalue()
			buffer.close()
			#plt.cla()
			plt.clf()

			graphic3 = base64.b64encode(image_png3)
			graphic3 = graphic3.decode('utf-8')

			#return render(request, 'progress.html',{'graphic':graphic3})



			fp = open("media/" + "result/" + str(id) + "/" + str(id) + ".log", 'a')
			fp.write("\nCLASSIFICATION DONE")
			fp.close()

			#context={'file':file,'graphic1':graphic1,'graphic2':graphic2,'graphic3':graphic3}
			list1=[a1,b1,c1]
			res=max(list1)

			fp = open("media/" + "result/" + str(id) + "/" + str(id) + ".log", 'r')
			file=""
			refresh =True
			for i in fp:
				if i == "CLASSIFICATION DONE":
					refresh = False
				file = "      "+ file +  "\n " + i + "\n"
			return render(request,"progress.html",context={'res':res,'file':file,'graphic1':graphic1,'graphic2':graphic2,'graphic3':graphic3})
		   

			#return HttpResponseRedirect("/progress")
		except Exception as e:
			print(e)
			print(request.FILES)
			return HttpResponse("""<h3> There was some error in our system. We will rectify it and will let you know.</h3>""")

	return render(request, 'decision.html');

'''def progress(request):
	fp = open("media/" + "result/" + str(id) + "/" + str(id) + ".log", 'r')
	file=""
	refresh =True
	for i in fp:
		if i == "CLASSIFICATION DONE":
			refresh = False
		file = "      "+ file +  "\n " + i + "\n"
	return render(request,"progress.html",{'file':file})'''









#from pyexpat import model
#from pyexpat import model
from math import comb
from flask import Flask , render_template , request
import matplotlib.pyplot as plt
import bgp
from bgp import reviewBook
import numpy
import sklearn


sorted_dict={}


app = Flask(__name__)

@app.route("/", methods = ['POST', 'GET'])
def hello():
    #temp1=[]
    if request.method == "POST":
        summary = request.form['summary']
        model=bgp.model
        genre_pred = bgp.reviewBook(summary,model)
        genre_label={0:"Children",1:"Classics",2:"Contemporary",3:"Fantasy",4:"Fiction",5:"Historical",6:"Historical Fiction",7:"Mystery",8:"Nonfiction",9:"Paranormal",10:"Romance",11:"Science Fiction",12:"Young Adult"}
        
        op=reviewBook(summary,model)
        comb={}
        per=[]
        x=[]
        y=[]
        for i in range(len(op[0])):
            #print(genre_label[i] , op[0][i])
            #per.append(op[0][i])
            x.append(op[0][i])
            y.append(genre_label[i])
            comb.update({genre_label[i]:op[0][i]})
        x[0]=op[0][0]*6
        print(comb)   
        plt.plot(x, y)
        plt.xlabel('predited value')
        plt.ylabel('genre')
        plt.title('Book genre prediction')
        plt.show()
        sorted_values = sorted(comb.values(), reverse=True) # Sort the values
        #temp1=op
        for i in sorted_values:
            for k in comb.keys():
                if comb[k] == i:
                    sorted_dict[k] = comb[k]
                    break
       
        
    return render_template("index.html", genre=sorted_dict)
"""
@app.route("/sub", methods = ['POST'])
def submit():
    
    #html to .py
    if request.method == "POST":
        name = request.form["username"]
    #.py to html
    return render_template("sub.html", n = name)
    """

if __name__ =="__main__":
    app.run()
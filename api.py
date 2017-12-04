from flask import Flask, redirect, url_for, request,render_template,jsonify
from main import process
import os

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
@app.route('/success')
def getImage():
      longtitude=request.args.get('longtitude')
      latitude=request.args.get('latitude')
      DIR = "{}_{}".format(longtitude, latitude)
      process(longtitude, latitude)
      candidates =[i for i in os.listdir(os.path.join('static',DIR)) if i[0].isdigit() ]
     
      # return render_template('result.html',image=os.path.join(DIR,'result.png' ),
                              # candidates = [os.path.join(DIR,i) for i in candidates])
      return jsonify([os.path.join('static',DIR,i) for i in candidates])

@app.route('/find',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      longtitude = request.form['longtitude']
      latitude = request.form['latitude']
      return redirect(url_for('getImage',longtitude = float(longtitude), latitude = float(latitude)))
   else:
      longtitude = request.args.get('longtitude')
      latitude = request.args.get('latitude')
      return redirect(url_for('getImage',longtitude = float(longtitude), latitude = float(latitude)))

if __name__ == '__main__':
   app.run(debug = True)
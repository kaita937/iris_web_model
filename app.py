from flask import Flask, render_template, request
from model.c45model import build_tree, predict, kategori

app = Flask(__name__)
tree = build_tree()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediksi():
    # Ambil input dari form
    sepalLength = float(request.form['sepalLength'])
    sepalWidth = float(request.form['sepalWidth'])
    petalLength = float(request.form['petalLength'])
    petalWidth = float(request.form['petalWidth'])

    # Ubah menjadi kategori
    sl = kategori(sepalLength, 4.3,5.5,5.6,6.7,6.8,7.9)
    sw = kategori(sepalWidth, 2.0,2.9,3.0,3.6,3.7,4.4)
    pl = kategori(petalLength, 1.0,3.3,3.4,5.6,5.7,7.0)
    pw = kategori(petalWidth, 0.1,0.8,0.9,1.6,1.7,2.5)

    query = {'SepalLengthCm': sl, 'SepalWidthCm': sw, 
             'PetalLengthCm': pl, 'PetalWidthCm': pw}

    hasil = predict(query, tree)

    if hasil == 'Iris-setosa' :
        image = 'iris_setosa.jpg'
    elif hasil == 'Iris-versicolor' :
        image = 'iris_versicolor.jpg' 
    else :
        image = 'iris_virginica.jpg' 

    return render_template('index.html', prediction=hasil, image=image)

if __name__ == '__main__':
    app.run(debug=True)

import tensorflow
import numpy

x = numpy.array([1,2,3,4,5],dtype=float)
y = numpy.array([2,4,6,8,10],dtype=float)

def custom_relu(x):
    return tensorflow.maximum(0.0, x)

model = tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.Dense(1,input_shape=[1],activation=custom_relu)
])

model.compile(optimizer="sgd",loss="mse")
model.fit(x,y,epochs=400)
print(model.predict(numpy.array([6])))
model.save("numprediction.h5")
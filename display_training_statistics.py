import matplotlib.pyplot as plot
import model_train

#plot the x-axis label
plot.xlabel("Epoch Number")

#plot the y-axis 
plot.ylabel("Amount of loss magnitude")

#can draw now.
#like: 
plot.plot(model_train.history.history['loss'])

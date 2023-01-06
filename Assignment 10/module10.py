from cats_vs_dogs import CatsDogs
from cats_vs_dogs_pre import CatsDogsPre

#calling the CatsDogs class
cats_vs_dogs = CatsDogs()
# Training the model
history = cats_vs_dogs.train()
#Ploting the Accuracy and loss graph
cats_vs_dogs.plot()
# Prediction of test images
cats_vs_dogs.predict()

#calling the CatsDogs class
cats_vs_dogs_pre = CatsDogsPre()
# Training the model
history_1 = cats_vs_dogs_pre.train()
#Ploting the Accuracy and loss graph
cats_vs_dogs_pre.plot()
# Prediction of test features
cats_vs_dogs_pre.predict()


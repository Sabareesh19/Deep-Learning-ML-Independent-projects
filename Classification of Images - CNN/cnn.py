# Convolutional Neural Network-used to classify images based on photographs

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras
#has tools for deep learning and CV and could import images.

# Part 1 - Building the CNN
#No need to encoding as the categorical variable is in pixels


# Importing the Keras libraries and packages
from keras.models import Sequential #to initialize the neural network. we can initialize through
#sequence of layers or as a graph
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense  #for the fully connected layer

# Initialising the CNN
classifier = Sequential() #initialize an object to method sequential

# Step 1 - Convolution
#Make sure there are no negative values in feature map
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
               #here 32 means there are 32 feature detectors of 3X3 dimensions
               #for input_shape = 64X64 for getting good accuracy results(64X64 is the dimensions of image coz we are using tensorflow)
   #activation is used to make it non linear and no -ve values to be added

# Step 2 - Pooling
#reduce the size of the feature map.Do with the slide of 2.Each slide takes a max               
classifier.add(MaxPooling2D(pool_size = (2, 2)))  #size of feature map divided by 2
 
# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))  #this is done to achieve more accuracy
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
#put all high values of spatial values to one single vector
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu')) #number around 100 is good practise. always take powers of 2.
#relu - according to how much it can pass to output
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#loss = binary_crossenthropy coz the ouput is either 0 or 1 and moreover the classifier is logistic regression



# Part 2 - Fitting the CNN to the images
#Image augumentation trains with less images and avoid overfitting

from keras.preprocessing.image import ImageDataGenerator

#image augumentation part
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True) #menas it will not repeat same image in other set

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', #from which directory we need to extract the images
                                                 target_size = (64, 64), #size of images expected by CNN model. i.e dimensions
                                                 batch_size = 32,
                                                 class_mode = 'binary')
#shortly called as svalidation generator
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,  #number of images in training set
                         nb_epoch = 25,  #number of epochs to train the CNN
                         validation_data = test_set, #on which set we need to evaluate the testing
                         nb_val_samples = 2000)  #number of images in the testing set


#accuracy on training set = 84%, testing test = 81%(not too bad)
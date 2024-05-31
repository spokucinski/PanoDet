import numpy as np
import keras
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import wordnet as wn
from keras import layers, callbacks
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import to_categorical
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('wordnet')

# tu generuje macierz podobienstwa
def generate_ImageNet_similarity_WordNet():
    offsets = []
    imagenet_classes = decode_predictions(to_categorical(np.expand_dims(np.array(range(1000)), axis=-1), num_classes=1000), top=1)

    for c in imagenet_classes:
        offsets.append(int(c[0][0].split('n')[1]))

    similarity_wordnet = np.zeros((1000, 1000))

    for i in range(1000):
        for j in range(1000):
            n = wn.synset_from_pos_and_offset('n', int(offsets[i]))
            m = wn.synset_from_pos_and_offset('n', int(offsets[j]))
            similarity_wordnet[i][j] = n.path_similarity(m)
    CSM_wordnet_nodiagonal = similarity_wordnet[~np.eye(similarity_wordnet.shape[0],dtype=bool)].reshape(similarity_wordnet.shape[0],-1)
    CSM_wordnet_nodiagonal = (CSM_wordnet_nodiagonal - np.min(CSM_wordnet_nodiagonal)) / (np.max(CSM_wordnet_nodiagonal) - np.min(CSM_wordnet_nodiagonal))
    return CSM_wordnet_nodiagonal

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# tu jakas siec dla imagenet
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 10

class MeanWeightsCallback(callbacks.Callback):
    def __init__(self):

        self.metric_values = []
        self.epochs = []
        self.similarities = generate_ImageNet_similarity_WordNet()
                
    def on_epoch_end(self, epoch, logs=None):

        # Get the weights of the last layer
        weights = self.model.layers[-1].get_weights()[0]  # Index 0 to get weights, 1 would be biases
        class_a = np.moveaxis(self.model.layers[-1].get_weights()[0], 0, -1)
        class_b = np.moveaxis(self.model.layers[-1].get_weights()[0], 0, -1)
        cs_tab_at_epoch = cosine_similarity(class_a, class_b)
        
        cs_tab_at_epoch_nodiagonal_raw = cs_tab_at_epoch[~np.eye(cs_tab_at_epoch.shape[0],dtype=bool)].reshape(cs_tab_at_epoch.shape[0],-1)
        cs_tab_at_epoch_nodiagonal = (cs_tab_at_epoch_nodiagonal_raw - np.min(cs_tab_at_epoch_nodiagonal_raw)) / (np.max(cs_tab_at_epoch_nodiagonal_raw) - np.min(cs_tab_at_epoch_nodiagonal_raw))
        
        cosine_value = np.sum(self.similarities*cs_tab_at_epoch_nodiagonal)/(np.sqrt(np.sum(self.similarities * self.similarities)) * np.sqrt(np.sum(cs_tab_at_epoch_nodiagonal * cs_tab_at_epoch_nodiagonal)))
        self.epochs.append(epoch)
        self.metric_values.append(cosine_value)
        print(f" - semantic similarity score: {cosine_value}")
  
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
mean_weights_callback = MeanWeightsCallback() 
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[mean_weights_callback])

SIZE = 15
sns.set_theme()
sns.set_style("darkgrid")
plt.rc('font', size=SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE - 3)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE - 3)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)    # legend fontsize
plt.rc('figure', titlesize=SIZE)  # fontsize of the figure title

plt.plot(mean_weights_callback.epochs, mean_weights_callback.metric_values, '#770122')
plt.xlabel("Epoch")
plt.ylabel("MAE between ES ans NS")
plt.savefig("WordNetImageNet - MAE between ES ans NS")
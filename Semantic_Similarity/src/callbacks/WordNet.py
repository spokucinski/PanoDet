import numpy as np
import keras
import matplotlib.pyplot as plt
import seaborn as sns

from keras import layers, callbacks
import nltk
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity


def get_offsets(class_names):  
    offsets = []
    nltk.download('wordnet')
    for word in class_names:
        # Find synsets for each word
        synsets = wn.synsets(word)
        # Get offsets for each synset
        word_offsets = [synset.offset() for synset in synsets]
        if len(word_offsets) > 1:
            offsets.append(word_offsets[0])
        else:
            offsets.append(word_offsets[0])
    return offsets
    
# tu generuje macierz podobienstwa
def generate_similarity_WordNet(class_names):  
    offsets = get_offsets(class_names)

    similarity_wordnet = np.zeros((len(offsets), len(offsets)))

    for i in range(len(offsets)):
        for j in range(len(offsets)):
            n = wn.synset_from_pos_and_offset('n', offsets[i])
            m = wn.synset_from_pos_and_offset('n', offsets[j])
            similarity_wordnet[i][j] = n.path_similarity(m)
    CSM_wordnet_nodiagonal = similarity_wordnet[~np.eye(similarity_wordnet.shape[0],dtype=bool)].reshape(similarity_wordnet.shape[0],-1)
    CSM_wordnet_nodiagonal = (CSM_wordnet_nodiagonal - np.min(CSM_wordnet_nodiagonal)) / (np.max(CSM_wordnet_nodiagonal) - np.min(CSM_wordnet_nodiagonal))
    return CSM_wordnet_nodiagonal

# Model / data parameters
num_classes = 10
input_shape = (32, 32, 3)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

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

checkpoint_filepath = "experiment_mnist/{epoch:02d}-{val_loss:.2f}.keras"
# model_checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq="epoch")

class MeanWeightsCallback(callbacks.Callback):
    def __init__(self):
        self.metric_values = []
        self.epochs = []
        # designed for a particular dataset - in this example - MNIST
        dataset = "CIFAR10" # "CIFAR10", "CIFAR100", "FMNIST"
        if dataset == "MNIST":
            self.classes = [
            "0",
            "1",     
            "2",     
            "3",     
            "4",       
            "5",         
            "6",      
            "7",        
            "8",    
            "9"
            ]
        elif dataset == "CIFAR10":
            self.classes = [
                "airplane",      
                "car",     
                "bird",      
                "cat",       
                "deer",        
                "dog",      
                "frog",      
                "horse",   
                "ship",     
                "truck" 
            ]
        elif dataset == "CIFAR100":
            self.classes = [
                "beaver", "dolphin", "otter", "seal", "whale",
                "fish", "flounder", "ray", "shark", "trout",
                "orchid", "poppy", "rose", "sunflower", "tulip",
                "bottle", "bowl", "can", "cup", "plate",
                "apple", "mushroom", "orange", "pear", "pepper",
                "clock", "keyboard", "lamp", "telephone", "television",
                "bed", "chair", "couch", "table", "wardrobe",
                "bee", "beetle", "butterfly", "caterpillar", "cockroach",
                "bear", "leopard", "lion", "tiger", "wolf",
                "bridge", "castle", "house", "road", "skyscraper",
                "cloud", "forest", "mountain", "plain", "sea",
                "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
                "fox", "porcupine", "possum", "raccoon", "skunk",
                "crab", "lobster", "snail", "spider", "worm",
                "baby", "boy", "girl", "man", "woman",
                "crocodile", "dinosaur", "lizard", "snake", "turtle",
                "hamster", "mouse", "rabbit", "shrew", "squirrel",
                "maple", "oak", "palm", "pine", "willow",
                "bicycle", "bus", "motorcycle", "truck", "train",
                "mower", "rocket", "streetcar", "tank", "tractor"
            ]
        elif dataset == "FMNIST":
            self.classes = [
                "t-shirt",        
                "trousers",     
                "sweater",      
                "dress",        
                "coat",        
                "sandal",      
                "shirt",       
                "sneakers",     
                "bag",          
                "boots"  
            ]

        self.similarities = generate_similarity_WordNet(self.classes)

                
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
plt.savefig("WordNet - MAE between ES ans NS")
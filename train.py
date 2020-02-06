from model import CNN
import torch

'''
    * Given data and labels, train model and store weights
    locally as file trained_cnn_weights.weights
    
    * The arguments data and labels default to 
        data = load_pkl("train_data.pkl"),
        labels = np.load("finalLabelsTrain.npy") if 
        not specified
    
    * The arguments data and labels are expected
    to be of the same type and structure as the 
    output of 
    
    load_pkl("train_data.pkl"),
    np.load("finalLabelsTrain.npy")
    
    respectively ( with variable imsage size ) . 
    
    I.E. : 
    
    data is a list of lists of lists:
    [ 
        [ [], ... , []  ],
        ... , 
        [ [], ... , []  ]  
    ]
        
    labels is a a numpy array of shape (6400,).
'''
def train( data=None, labels=None ):
    weights_file_path = "./trained_cnn_weights.weights"
    network = CNN(lr=0.001, batch_size=64, epochs=15, number_of_classes=9)
    network._train( data, labels )
    torch.save(network.state_dict(), weights_file_path )
    return None



# driver for convinience,
# will read data and labels as arguments from sys.argv
# where data_filename = sys.argv[1] and labels_filename = sys.argv[2], and will exeucute default hehavior if
# neither are specified.
import sys

# Given that this is an unrequired, for-convienience driver, no guarantees are extended for invalid arguments
if __name__ == "__main__":
    data = None
    labels = None

    if ( len(sys.argv) > 2 ):
        data = load_pkl(sys.argv[1])
        labels = load_pkl(sys.argv[2])

    train( data, labels )

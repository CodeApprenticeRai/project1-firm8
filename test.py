from model import CNN
import loaddata
import torch, numpy as np

'''
    * Given a collection of images 
    return an output vector of predicted labels
    for each image as estimated by the CNN module found
    in model.py
    
    * The argument data is expected
    to be of the same type and structure as the 
    output of load_pkl("train_data.pkl") ( with variable imsage size ) ,

    I.E. : 
    
    data is a list of lists of lists:
    [ 
        [ [], ... , []  ],
        ... , 
        [ [], ... , []  ]  
    ]
'''
def test( data=None ):
    weights_file_path = "./pre_trained_cnn_weights.weights"
    network = CNN(lr=0.001, batch_size=8, epochs=15, number_of_classes=9)
    weights = torch.load(weights_file_path)
    network.load_state_dict(weights)

    if ( data == None ):
        data = loaddata.load_pkl("train_data.pkl")

    data = [ [np.array(data[i], dtype=bool)] for i in range(len(data))]
    data = network.standardize_data( data )
    data = torch.stack( [ data[i][0] for i in range(len(data)) ] )

    return network.predict( data )



# driver for convinience,
# will read data and labels as arguments from sys.argv
# where data_filename = sys.argv[1]
import sys

# Given that this is an unrequired, for-convienience driver, no guarantees are extended for invalid arguments
if __name__ == "__main__":
    data = None

    if ( len(sys.argv) > 1 ):
        data = loaddata.load_pkl(sys.argv[1])

    test( data )

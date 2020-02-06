import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from skimage.transform import resize
import pickle, numpy as np
import loaddata


class CNN(nn.Module):
    def __init__(self, lr, epochs, batch_size, number_of_classes):
        super(CNN, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.number_of_classes = number_of_classes
        self.loss_history = []
        self.accuracy_history = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.bn6 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)

        input_dimensions = self.calculate_input_dimensions()

        self.fc1 = nn.Linear( input_dimensions, self.number_of_classes )

        self.optimizer = torch.optim.Adam( self.parameters(), lr=self.lr )
        self.loss = nn.CrossEntropyLoss()

        self.to(self.device)

        self.prepare_data()


    def calculate_input_dimensions( self ):
        batch_data = torch.zeros( (1, 1, 54, 54) )
        batch_data = self.conv1(batch_data)
        batch_data = self.conv2(batch_data)
        batch_data = self.conv3(batch_data)
        batch_data = self.maxpool1(batch_data)
        batch_data = self.conv4(batch_data)
        batch_data = self.conv5(batch_data)
        batch_data = self.conv6(batch_data)
        batch_data = self.maxpool2(batch_data)

        return int( np.prod( batch_data.size() ) )

    def forward( self, batch_data ):
        batch_data = torch.tensor( batch_data ).to( self.device )

        batch_data = self.conv1(batch_data)
        batch_data = self.bn1(batch_data)
        batch_data = torch.nn.functional.relu(batch_data)

        batch_data = self.conv2(batch_data)
        batch_data = self.bn2(batch_data)
        batch_data = torch.nn.functional.relu(batch_data)

        batch_data = self.conv3(batch_data)
        batch_data = self.bn3(batch_data)
        batch_data = torch.nn.functional.relu(batch_data)

        batch_data = self.maxpool1(batch_data)

        batch_data = self.conv4(batch_data)
        batch_data = self.bn4(batch_data)
        batch_data = torch.nn.functional.relu(batch_data)

        batch_data = self.conv5(batch_data)
        batch_data = self.bn5(batch_data)
        batch_data = torch.nn.functional.relu(batch_data)

        batch_data = self.conv6(batch_data)
        batch_data = self.bn6(batch_data)
        batch_data = torch.nn.functional.relu(batch_data)

        batch_data = self.maxpool2(batch_data)

        batch_data = batch_data.view( batch_data.size()[0], -1 )

        classes = self.fc1( batch_data )
        return classes

    def standardize_data( self, data ):
        for i in range(len(data)):
            #try:
            data[i][0] = torch.Tensor( resize( data[i][0], (54, 54) ).astype(np.float64) ).unsqueeze(0)
            #except:
                #continue

        return data

    def prepare_data( self, data=None, labels=None):

        if ( data == None ):
            data = loaddata.load_pkl("train_data.pkl")
            labels = np.load("finalLabelsTrain.npy")

        data = np.array( [ np.array( data[i], dtype=bool ) for i in range(len(data)) ] )
        data_labelled =  [ [data[i], labels[i]] for i in range(len(data)) ]

        data_labelled_standardized = self.standardize_data( data_labelled )
        # np.random.shuffle( std_labelled_training_data )

        self.training_data_loader = torch.utils.data.DataLoader(
            data_labelled_standardized[ : int( len(data_labelled_standardized) * 0.80 ) ],
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=0)

        self.test_data_loader = torch.utils.data.DataLoader(
            data_labelled_standardized[ int( len(data_labelled_standardized) * 0.80 ) : ],
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=0)
        return None


    def _train( self, data=None, labels=None ):
        print( "Batch Size: {}".format(self.batch_size))
        self.train()

        if ( data != None ): # if training-data passed, override default data configuration
            self.prepare_data( data, labels )


        for i in range(self.epochs):
            epoch_loss = 0
            epoch_accuracy = []

            for j, (input, label) in enumerate(self.training_data_loader):
                self.optimizer.zero_grad()
                label = label.to(self.device)
                prediction = self.forward(input)
                loss = self.loss(prediction, label.long())
                prediction = torch.nn.functional.softmax(prediction, dim=1)
                classes = torch.argmax(prediction, dim=1)

                wrong = torch.where( classes != label,
                                    torch.tensor([1.]).to(self.device),
                                    torch.tensor([0.]).to(self.device)
                    )

                accuracy = 1 - torch.sum(wrong) / self.batch_size

                epoch_accuracy.append( accuracy.item() )
                self.accuracy_history.append( accuracy.item() )

                epoch_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            performance_summary = "Finished epoch {}. Total Loss: {}, Accuracy: {} ".format( i, epoch_loss, np.mean( epoch_accuracy) )
            print( performance_summary  )

            self.loss_history.append( epoch_loss )
        return None

    def _test( self ):
        self.eval()

        epoch_loss = 0
        epoch_accuracy = []


        for j, (input, label) in enumerate(self.test_data_loader):
            label = label.to(self.device)
            prediction = self.forward(input)
            loss = self.loss(prediction, label.long())
            prediction = torch.nn.functional.softmax(prediction, dim=1)
            classes = torch.argmax(prediction, dim=1)

            wrong = torch.where(classes != label,
                                torch.tensor([1.]).to(self.device),
                                torch.tensor([0.]).to(self.device)
                                )

            accuracy = 1 - torch.sum(wrong) / self.batch_size

            epoch_accuracy.append( accuracy.item() )
            self.accuracy_history.append( accuracy.item() )

            epoch_loss += loss.item()


        print("Total Loss: {}, Accuracy: {} ".format( epoch_loss, np.mean(epoch_accuracy) ) )
        return None

    def predict( self, X, visualize=False ):
        self.eval()

        with torch.no_grad():
            prediction = self.forward(X)
            prediction = torch.nn.functional.softmax(prediction, dim=1)
            classes = torch.argmax(prediction, dim=1)

            return classes



from os import path
import time

if __name__ == "__main__":
    weights_file_path = "./pre_trained_cnn_weights.weights"
    network = CNN( lr=0.001, batch_size=64, epochs=15, number_of_classes=9 )

    if not path.exists(weights_file_path):
        t = time.time()
        network._train()
        t = time.time() - t
        print( "Network Traning Time in Seconds: {} ".format(t) )
        torch.save(network.state_dict(), weights_file_path)

    weights = torch.load(weights_file_path)
    network.load_state_dict(weights)

    plt.title("Error vs Number of Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.plot(network.loss_history)
    plt.show()
    plt.savefig("error_history.png")

    plt.title("Accuracy vs Number of Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.plot(network.accuracy_history)
    plt.show()

    network._test()

import matplotlib.pyplot as plt
import numpy as np
import pickle

def get_data():
    """This function reads the data from CSV file and separates the data
     to Training, Validation and Test data
    """


    X_train,Y_train,X_val,Y_val,X_test,Y_test=[],[],[],[],[],[]

    dict={'Training\n':[Y_train,X_train],
          'PublicTest\n': [Y_val, X_val],
          'PrivateTest\n': [Y_test, X_test]}
    file=open('../facial_recog_data/fer2013.csv')

    for line in file:
        row = line.split(',')
        if row[-1] in dict:
            y,x=dict[row[2]]
            y.append(int(row[0]))
            x.append([int(i) for i in row[1].split(' ')])


    return X_train,Y_train,X_val,Y_val,X_test,Y_test


def y2indicator(y):
    N=len(y)
    ind=np.zeros((N,10))         #one hot encoding

    for i in range(N):
        ind[i,y[i]]=1
    return ind



def normalize_data(Xtrain,Xtest,Xval):

    #Calculate mean and std
    train_mean = np.mean(Xtrain, axis=0)
    train_std = np.std(Xtrain, axis=0)

    #normalize the data
    Xtrain = (Xtrain - train_mean) / train_std
    Xtest = (Xtest - train_mean) / train_std
    Xval = (train_mean - Xval) / train_std

    return Xtrain,Xtest,Xval





def rearrange_data(X):
    N=len(X)
    X_img=np.zeros(shape=[N,48,48],dtype=np.float32)
    for i in range(N):
        X_img[i,:,:]=np.array(X[i]).reshape(48,48)

    return X_img





if __name__=="__main__":

    #get the data
    X_train_flat, Y_train, X_val_flat, Y_val, X_test_flat, Y_test=get_data()





    #Reshape to get images
    X_train=rearrange_data(X_train_flat)
    X_val=rearrange_data(X_val_flat)
    X_test=rearrange_data(X_test_flat)

    #encode the output
    Y_train_ind=y2indicator(Y_train)
    Y_val_ind=y2indicator(Y_val)
    Y_test_ind=y2indicator(Y_test)



    #normalize data
    X_train,X_test,X_val=normalize_data(Xtrain=X_train,Xtest=X_test,Xval=X_val)


    #pickle data
    pickle.dump((X_train, Y_train, Y_train_ind), open('../facial_recog_data/train_data.p', "wb"))
    pickle.dump((X_test, Y_test, Y_test_ind), open('../facial_recog_data/test_data.p', "wb"))
    pickle.dump((X_val, Y_val, Y_val_ind), open('../facial_recog_data/val_data.p', "wb"))

    # im = plt.imshow(X_train[10, :, :], cmap='gray')
    # plt.show()


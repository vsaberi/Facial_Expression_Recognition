import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import time
from sklearn.utils import shuffle
import pickle
from datetime import datetime
import os






def W_variable(name,shape,type="fc"):
    """"Returns convolutional layer weight. If the name already exists it just retrieves it
    type= "conv" or "fc"
    """
    if type=="conv":
        w=tf.get_variable(name=name,shape=shape,dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32))
    elif type=="fc":
        w= tf.get_variable(name=name,shape=shape,dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    return w


def b_variable(shape):
    """"Returns a bias variable
    """
    return tf.Variable(tf.constant(0.0,shape=shape))


class Conv_pool_layer:

    def __init__(self,
                 layer_name,                #layer name
                 size,                      #size (w,h,num_input_channel,num_output_channel)
                 pooling_size,              #max pooling layer size (w,h). "None" removes pooling layer
                 activation,                #options: "sigmoid", "relu", "None"
                 ):

        self.layer_name=layer_name
        self.size=size
        self.pooling_size=pooling_size
        self.activation=activation


        #create weight
        b_shape=[size[-1]]
        self.w=tf.get_variable(name=layer_name,shape=size,dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32))
        self.b=tf.get_variable(name=layer_name+'_b',dtype=tf.float32,initializer=np.zeros(shape=b_shape,dtype=np.float32))


    def forward(self,
                input,
                dropout_keep_prob=None               # dropout keep probablity (None: no dropout)
                ):
        #output operation
        output=tf.nn.conv2d(input=input,
                             filter=self.w,
                             strides=[1,1,1,1],
                             padding="SAME")

        #apply bias
        output+=self.b

        #activation
        if self.activation == "relu":
            output = tf.nn.relu(output)
        elif self.activation == "sigmoid":
            output = tf.nn.sigmoid(output)


        #pooling (down-sampling)
        if self.pooling_size is not None:
            output=tf.nn.max_pool(value=output,
                                  ksize=[1,self.pooling_size[0],self.pooling_size[1],1],
                                  strides=[1,self.pooling_size[0],self.pooling_size[1],1],
                                  padding="SAME")
        if dropout_keep_prob is not None:
            output=tf.nn.dropout(output,keep_prob=dropout_keep_prob)

        return output





class Fully_conn_layer:

    def __init__(self,
                 layer_name,                    #layer name
                 size,                          #size (num_inputs,num_outputs)
                 activation="relu",       #options: "sigmoid", "relu", "None"
            ):

        self.layer_name=layer_name
        self.size=size
        self.activation=activation



        #create weight
        b_shape=[size[-1]]
        self.w= tf.get_variable(name=layer_name,shape=size,dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        self.b=tf.get_variable(name=layer_name+'_b',shape=b_shape,dtype=tf.float32,initializer=tf.constant(0.0,shape=b_shape))



    def forward(self,
                input,
                dropout_keep_prob = None            # dropout keep probablity (None: no dropout)
                ):

        #output operation
        output=tf.matmul(input,self.w)

        output+=self.b


        #pooling (down-sampling)
        if self.activation=="relu":
            output=tf.nn.relu(output)
        elif self.activation=="sigmoid":
            output=tf.nn.sigmoid(output)


        if dropout_keep_prob is not None:
            output=tf.nn.dropout(output,keep_prob=dropout_keep_prob)

        return output







class CNN:

    def __init__(self,
                 conv_layers_size,
                 max_pooling_layers_size,
                 conv_dropout_keep_prob,
                 conv_activation,
                 hidden_layers_size,
                 hidden_activation,
                 hidden_dropout_keep_prob):

        self.conv_layers_size=conv_layers_size
        self.max_pooling_layers_size=max_pooling_layers_size
        self.hidden_layers_size=hidden_layers_size
        self.conv_activation=conv_activation
        self.conv_dropout_keep_prob=conv_dropout_keep_prob
        self.hidden_activation=hidden_activation
        self.hidden_dropout_keep_prob=hidden_dropout_keep_prob




    def _create_layers(self,
                       num_input_channel,               #input image number of channels
                       num_classes                      #numer of classification classes
                       ):

        # create conv layers
        self.conv_pool_layers = []

        for i, layer_sz in enumerate(self.conv_layers_size):
            layer = Conv_pool_layer(layer_name='conv' + str(i),
                                    size=[layer_sz[0], layer_sz[1], num_input_channel, layer_sz[-1]],
                                    pooling_size=self.max_pooling_layers_size[i],
                                    activation=self.conv_activation[i],
                                    )

            self.conv_pool_layers.append(layer)
            num_input_channel = layer_sz[-1]

        # create hidden layers
        num_inputs = np.prod(self.conv_layers_size[-1])
        self.fc_layers = []
        for i, layer_sz in enumerate([self.hidden_layers_size, num_classes]):
            layer = Fully_conn_layer(layer_name='hd' + str(i),
                                     size=[num_inputs, layer_sz],
                                     activation=self.hidden_activation[i],
                                     )

            self.fc_layers.append(layer)
            num_inputs = layer_sz

    def _forward(self,X,conv_dropout_keep_prob,hidden_dropout_keep_prob):

        output=X

        #conv_layers
        for i,layer in enumerate(self.conv_pool_layers):
            output=layer.forward(output,conv_dropout_keep_prob[i])

            if self.conv_dropout_keep_prob[i] is not None:
                output = tf.nn.dropout(output, keep_prob=self.conv_dropout_keep_prob[i])

        #flatten the output
        output=self._flatten(output)

        #fc layers
        for i,layer in enumerate(self.fc_layers):
            output=layer.forward(output,[hidden_dropout_keep_prob,])
            if self.hidden_dropout_keep_prob[i] is not None:
                output = tf.nn.dropout(output, keep_prob=self.hidden_dropout_keep_prob[i])

        return output






    def _flatten(self,input):
        """This function flattens the convolutional leyrs output to feed to fully-connected layer
        """

        # find the shape
        shape = input.get_shape()

        num_features = np.prod(shape[1:])

        # reshape
        output = tf.reshape(input, [-1, num_features])

        return output, num_features



    def fit(self,X,Y_ind,X_val,Y_ind_val,num_iterations,batch_sz,print_period):

        #get input and output dimensions
        N,width,height,num_input_channel=X.shape
        _,num_classes=Y_ind.shape

        #create layers
        self._create_layers(num_input_channel=num_input_channel,
                            num_classes=num_classes
                            )

        # define placeholders
        x = tf.placeholder(tf.float32, shape=(None, width, height, num_input_channel), name='x')
        y_ind = tf.placeholder(tf.float32, shape=(None, num_classes), name='y')
        # y_ind = tf.cast(tf.argmax(y, dimension=1), dtype=tf.float32)
        # keep = tf.placeholder(tf.float32, name='keep')


        # model output
        logits = self._forward(X=x,conv_dropout_keep_prob=self.conv_dropout_keep_prob,hidden_dropout_keep_prob=self.hidden_dropout_keep_prob)

        # training optimization operation with decaying learning rate
        cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.96, staircase=True)
        train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost, global_step=global_step)

        # accuracy calculation operation
        logits_predict = self._forward(X=x, conv_dropout_keep_prob=len(self.conv_dropout_keep_prob)*[None],hidden_dropout_keep_prob=len(self.hidden_dropout_keep_prob)*[None])
        Y_ind_prediction = tf.cast(tf.argmax(logits_predict, dimension=1),tf.float32)

        accuracy_op = tf.reduce_mean(tf.cast(tf.equal(Y_ind_prediction, y_ind), tf.float32))



        session = tf.Session()
        session.run(tf.initialize_all_variables())



        for i in range(num_iterations):

            # set batch data
            offset = (i * batch_sz) % (X.shape[0] - batch_sz)
            X_batch = X[offset:(offset + batch_sz), :, :, :]
            Y_ind_batch = Y_ind[offset:(offset + batch_sz), :]

            # train
            session.run(train_op, feed_dict={x: X_batch, y_ind: Y_ind_batch})

            if i % print_period == 0:
                batch_acc = session.run(accuracy_op, feed_dict={x: X_batch, y_ind: Y_ind_batch})
                print('batch accuracy at step %d:%.4f' % (i, batch_acc))

                val_acc = session.run(accuracy_op, feed_dict={x: X_val, y_ind: Y_ind_val})
                print('validation accuracy at step %d:%.4f' % (i, val_acc))

                # save session
                tf.train.Saver().save(session, 'saved_sessions/', global_step=global_step)



























def main():


    #load data
    Xtrain, Ytrain, Ytrain_ind = pickle.load(open("../facial_recog_data/train_data.p", "rb"))
    Xtest, Ytest, Ytest_ind = pickle.load(open("../facial_recog_data/test_data.p", "rb"))
    Xval, Yval, Yval_ind = pickle.load(open("../facial_recog_data/val_data.p", "rb"))





    model=CNN(conv_layers_size=[(5,5,20),(5,5,40)],
              max_pooling_layers_size=[(2,2),None],
              conv_dropout_keep_prob=[0.5, 1.0],
              conv_activation=['relu','relu'],
              hidden_layers_size=[256,50],
              hidden_activation=['relu','relu'],
              hidden_dropout_keep_prob=[1.0,1.0])

    model.fit(Xtrain,Ytrain_ind,Xval,Yval_ind,5000,100,10)










if __name__=="__main__":
    main()






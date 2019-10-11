import tensorflow as tf
import pandas as pd
import seaborn as sb
import numpy as np
​
x=tf.placeholder(tf.float32,shape=(None,),name="x")
y=tf.placeholder(tf.float32,shape=(None,),name="y")
W=tf.Variable(np.random.normal(),name="weight")
b=tf.Variable(np.random.randn(),name="bias")
y_pred=tf.add(tf.multiply(x,W),b)
#cost function
cost=tf.reduce_mean(tf.square(y_pred- y))
​
#randomly generating data
x_batch=np.linspace(-1,1,101)
y_batch=2*x_batch + np.random.randn(*x_batch.shape)*0.3
lr=0.2
optimizer=tf.train.AdamOptimizer(lr).minimize(cost)
init=tf.global_variables_initializer()
with tf.Session() as session:
  session.run(init)
with tf.Session() as session:
  session.run(init)
  feed_dict={x:x_batch,y:y_batch}
  for _ in range(40):
    loss_var,_=session.run([cost,optimizer],feed_dict)
    print("loss=",loss_var)
    y_pred_batch=session.run(y_pred,{x:x_batch})
​
​
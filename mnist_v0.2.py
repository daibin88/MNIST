import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data

mnist = input_data.read_data_sets("data/",one_hot=True)

sess = tf.InteractiveSession()

x=tf.placeholder("float",[None,784])
y_=tf.placeholder("float",[None,10])

# 权重初始化
## 定义一个函数，用于初始化所有的权重值 W
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

## 定义一个函数，用于初始化所有的偏置项
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

## 卷积与池化

def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],
                         strides=[1,2,2,1],padding = 'SAME')

#第一层卷积
## 由一个卷积接一个max pooling完成，卷积在每个5x5的patch中算出32个特征。
## 卷积的权重张量形状是[5,5,1,32],前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目
## 对于每一个输出通道都有一个对应的偏置量

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

#将输入图像x变为4d向量，第2维，第3维对应图片的宽、高，最后一维代表图片的颜色通道数，灰度图像为1，rgb图像为3

x_image = tf.reshape(x,[-1,28,28,1])
#x_image = tf.reshape(x,[-1,28,28,1],name=None)

# 将x_image 和权值向量进行卷积，加上偏置项，然后运用ReLu激活函数，最后进行max_pooling
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

###-------------------------
## 第二层卷积
## 每个patch会得到64个特征
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

###----------------------------

## 全连接层

##现在图片大小减小为7*7，加入一个有1024个神经元的全连接层，用于处理整个图片
## 将池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

## Dropout 
##为了减少过拟合，在输出层之前加入dropout。利用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率
## 在训练过程中启用dropout，在测试过程中关闭dropout。
## TensorFlow的tf.nn.dropout 操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

## 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

## 根据交叉熵公式求值
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
## 利用tf自带的求交叉熵方法
#y_conv_t=tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv_t,labels=y_))
### 训练和评估模型
## 使用更加复杂的ADAM优化器来做梯度最速下降，在feed_dict中加入额外的参数keep_prob 来控制dropout比例，
## 并且每100次迭代输出一次日志
#y_resolut=tf.nn.softmax(y_conv_t)

### y_conv：预测值，y_:真值

train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
sess.run(tf.global_variables_initializer())

for i in range(2000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob: 1.0})
        print("step %d, training accuracy %g" %(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob: 0.5})
    
print("test accuracy %g"%accuracy.eval(feed_dict={
    x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
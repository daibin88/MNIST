import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
import os

mnist = input_data.read_data_sets("data/",one_hot=True)

"""实现回归模型"""
## 通过操作符号变量来描述这些课交互的操作单元
## x不是一个值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。
## 希望能够输入任意的MNIST图像，每一张图像展平成784维的向量。
## W和b为权重值和偏置量，使用Variable定义，代表可修改的张量
#定义变量
x=tf.placeholder("float",[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

#定义模型
y = tf.nn.softmax(tf.matmul(x,W) + b) #定义我们的模型，预测值

#定义损失函数
y_ = tf.placeholder("float",[None,10]) # 计算交叉熵,真值
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 利用梯度下降法以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化创建的变量
init = tf.global_variables_initializer()

#模型保存
saver = tf.train.Saver()
model_name = "model.ckpt"
model_dir = "%s" % ("test1") #将srcnn和label_size=21，组成一个srcnn_21的文件名
checkpoint_dir = os.path.join('checkpoint', model_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

sess = tf.Session()
sess.run(init)
# 下面循环，随机抓取训练数据中的100个批处理数据点，用这些数据点作为参数替换之前的占位
#符来运行 train_step,此处使用的随机梯度下降训练

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict = {x:batch_xs, y_:batch_ys })

saver.save(sess,os.path.join(checkpoint_dir, model_name))
"""评估模型"""

# argmax 给出某个tensor对象在某一维数据上的器数据最大值所在的索引值
 
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float")) # cast将布尔型与float型之间相互转化

"""准确率计算"""
print(sess.run(accuracy,feed_dict={x: mnist.test.images,y_:mnist.test.labels}))
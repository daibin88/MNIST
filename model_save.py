# 模型保持和加载
#
import tensorflow as tf
import numpy as np
import os

x = tf.placeholder(tf.float32,shape=[None,1])
y = 4 *x + 4

w = tf.Variable(tf.random_normal([1],-1,1))
b = tf.Variable(tf.zeros([1]))
y_predict = w*x + b

loss = tf.reduce_mean(tf.square(y - y_predict))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)


saver = tf.train.Saver()
x_data = np.reshape(np.random.rand(10).astype(np.float32),(10,1))

def main(isTrain):
    train_steps = 400
    checkpoint_steps = 50
    checkpoint_dir = ''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        model_name = "model.ckpt"
        model_dir = "%s" % ("test3") #将srcnn和label_size=21，组成一个srcnn_21的文件名
        checkpoint_dir = os.path.join('checkpoint', model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        if isTrain:
            for i in range(0,train_steps):
                _, err = sess.run([train,loss],feed_dict={x:x_data})
                if(i+1)% checkpoint_steps ==0:
                    print("setp: [%2d], loss; [%.6f]" %(i+1, err))
                
                    saver.save(sess,os.path.join(checkpoint_dir, model_name),global_step = i+1)
        else:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir) #获取保存的模型
            print(ckpt.model_checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path: #model_checkpoint_path获取最新的模型
                saver.restore(sess,ckpt.model_checkpoint_path)
            else:
                pass
            print(sess.run(w))
            print(sess.run(b))
if __name__ == '__main__':
    main(True)
    main(False)

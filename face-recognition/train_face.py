
# coding: utf-8

# In[46]:


import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split


# In[47]:


my_face_path = '/home/weifeng/learngit/tf/人脸识别/my_faces'
other_face_path = '/home/weifeng/learngit/tf/人脸识别/other_face'
size = 64
imgs = []
labs = []


# In[48]:


def getpaddingSize(img):
    h,w,_ = img.shape
    top,bottom,left,right = (0,0,0,0)
    longest = max(h,w)
    if w < longest:
        tmp = longest - w
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top,bottom,left,right


# In[49]:


def readdata(path,h=size,w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            img = cv2.imread(filename)
            top,bottom,left,right  =getpaddingSize(img)
            # 将图片放大,扩充图片边缘部分
            img = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value = [0,0,0])
            img = cv2.resize(img,(h,w))
            imgs.append(img)
            labs.append(path)
    


# In[50]:


readdata(my_face_path)
readdata(other_face_path)


# In[51]:


# 将图片数据与标签变成数组
imgs = np.array(imgs)
labs = np.array([[0,1] if lab == my_face_path  else [1,0] for lab in labs])


# In[52]:


#随机划分测试集和训练集
train_x,test_x,train_y,test_y = train_test_split(imgs,labs,test_size = 0.05,random_state = random.randint(0,100))
#参数:图片的总数,图片的 高 宽 通道,其实可以不写
train_x = train_x.reshape(train_x.shape[0],size,size,3)
test_x = test_x.reshape(test_x.shape[0],size,size,3)


# In[57]:


#归一化
trian_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0
#生成 picture_batch 最好是2的指数
batch_size = 100
num_batch = len(train_x) // batch_size
# tf.placeholder(dtype,[row,shape])
x = tf.placeholder(tf.float32,[None,size,size,3])
y = tf.placeholder(tf.float32,[None,2])
keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)


# In[56]:





# In[30]:


def weightVariable(shape):
    init = tf.random_normal(shape,stddev=0.01)
    return tf.Variable(init)


# In[31]:


def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)


# In[32]:


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = 'SAME')


# In[33]:


def maxPool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding = 'VALID')


# In[34]:


def dropout(x,keep):
    return tf.nn.dropout(x,keep)


# In[ ]:


def cnnLayer():
    #第一层 卷积核大小(3x3) 通道 3 过滤器个数 32
    W1 = weightVariable([3,3,3,32])
    b1 = biasVariable([32])
    #卷积
    co


# In[38]:


def cnnLayer():
    #第一层 卷积核大小(3x3) 通道 3 过滤器个数 32
    W1 = weightVariable([3,3,3,32])
    b1 = biasVariable([32])
    #卷积
    conv1 = tf.nn.relu(conv2d(x,W1)+b1)
    #池化
    pool1 = maxPool(conv1)
    #通过 dropout使某些权重不更新,以减少过拟合
    drop1 = dropout(pool1,keep_prob_5)
    
    #第二层
    W2 = weightVariable([3,3,32,64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1,W2)+b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2,keep_prob_5)
    
    #第三层
    W3 = weightVariable([3,3,64,64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2,W3)+b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3,keep_prob_5)
    
    #全连接层
    #每次pool h,w 会变成一半 所以 x.shape = [8,8,64] 自定义节点个数为512
    Wf = weightVariable([8*8*64,512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3,[-1,8*8*64])  # 展开成一行
    dense = tf.nn.relu(tf.matmul(drop3_flat,Wf)+bf) # 1 * 512
    dropf = dropout(dense,keep_prob_75)
    
    # 输出层
    Wout = weightVariable([512,2])
    bout = biasVariable([2])
    out = tf.add(tf.matmul(dropf,Wout),bout)
    
    return out


# In[66]:


def cnnTrain():
    out = cnnLayer()
    #交叉熵
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out,labels = y))
    train_step  = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    #比较标签是否相等,再求所有数的平均值
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out,1),tf.argmax(y,1)),tf.float32))
    # 将loss 和 accuracy 保存供 tensorboard使用
    tf.summary.scalar('loss',cross_entropy)
    tf.summary.scalar('accuracy',accuracy)
    merged_summary_op = tf.summary.merge_all()
    # 数据保存器的初始化
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('/home/weifeng/learngit/tf/graph',graph = tf.get_default_graph())
        
        for n in range(10):
            # batch_size = 32
            for i in range(num_batch):
                batch_x = train_x[i*batch_size:(i+1)*batch_size]
                batch_y = train_y[i*batch_size:(i+1)*batch_size]
                
                # 开始训练,同时训练3变量,返回3个数据
                
                _,loss,summary = sess.run([train_step,cross_entropy,merged_summary_op],feed_dict={x:batch_x,y:batch_y,keep_prob_5:0.5,keep_prob_75:0.75})
                summary_writer.add_summary(summary,n*num_batch+i)
                
                #打印loss
#print(n*num_batch+i,loss)
                
                if (n*num_batch+i) % 50 == 0 :
                    #获取测试数据的准确率  eval 一次获得一个tensor中的值,但是run可以获得多个tensor的值
                    acc = accuracy.eval({x:test_x, y:test_y, keep_prob_5:1.0, keep_prob_75:1.0})
                    print(n*num_batch+i,acc)
                    if acc > 0.96 and n>2:
                        saver.save(sess,'/home/weifeng/learngit/tf/人脸识别/train_faces.model',global_step = n*num_batch+i)
                        sys.exit(0)
        print('accuracy less 0.98,handsome!')


# In[67]:


cnnTrain()


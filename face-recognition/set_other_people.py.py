
# coding: utf-8

# In[1]:


import sys
import cv2
import os
import dlib


# In[2]:


input_dir = '/home/weifeng/桌面/oter'
output_dir = '/home/weifeng/learngit/tf/人脸识别/other_face1'
size = 64
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# # 使用 dlib 的frontal_face_detector 作为特征提取器

# In[3]:


detector = dlib.get_frontal_face_detector()

index = 1
for (path,dirname,filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Being processed picture %s' % index)
            img_path = path+'/'+filename
            # 从文件中读取照片
            img = cv2.imread(img_path)
            #转换为灰度照片
            gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #使用detector进行人脸检测 dets 为返回结果
            dets = detector(gray_img,1)
            #使用enumerate函数遍历序列中的元素以及他们的下标
            #下标i即为人脸序号
            #left: 人脸左边距离图片左边界的距离
            #right: 人脸右边距离图片左边界的距离
            #top:人脸上边界距离图片上边界的距离
            #bottom: 人脸下边界距离上边界的距离
            for i , d  in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left()>0 else 0 
                y2 = d.right() if d.right() > 0 else 0
                face = img[x1:y1,x2:y2]
                #调整图片的尺寸
                face = cv2.resize(face,(size,size))
                cv2.imshow('image',face)
                #保存照片
                cv2.imwrite(output_dir+'/'+str(index)+'.jpg',face)
                index +=1
            key = cv2.waitKey(30) & 0xff
            if key == 'q':
                sys.exit(0)


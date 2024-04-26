import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pywt
import cv2
import random
import math

#载入水印与原图
watermark = np.array(Image.open('water_mark/watermark.png'))#获取水印像素数组
row, col = watermark.shape[:2]  #水印真实shape为(64,64,4)
img = np.array(Image.open('origin_img/Lena.bmp'))#获取待添加水印图片像素数组

def arnold(mark):   #水印arnold置乱处理
    p = np.zeros((row, col, 4), np.uint8)   #用于储存变换后的像素数组
    for i in range(row):
        for j in range(col):
            x = (i + j) % row
            y = (i + 2 * j) % col
            p[x, y] = mark[i, j]
    return p   #返回置乱过后的像素数组

def haar(img):#haar变换并返回四部分拼好的图像
    coeffs = pywt.dwt2(img, 'haar')#使用pywt库的二维小波一级变换
    cA, (cH, cV, cD) = coeffs
    AH = np.concatenate([cA, cH], axis=1)
    VD = np.concatenate([cV, cD], axis=1)
    img = np.concatenate([AH, VD], axis=0)
    return img

def enmark(img,mark):#设置水印注入规则，对象是划分后的像素块阵
    k=25#嵌入强度，应保证调整后所选位置像素点像素差大于k
    for i in range(64):
        for j in range(64):
            if mark[i][j]==255:#如果水印对应像素点为白，则设置(2,2)位置像素值大于(2,3)
                if img[i][j][2,2]<img[i][j][2,3]:
                    t=img[i][j][2,2]
                    img[i][j][2,2]=img[i][j][2,3]
                    img[i][j][2,3]=t
                    if img[i][j][2,2]-img[i][j][2,3]<k:#符合嵌入强度
                        img[i][j][2,2]=img[i][j][2,2]+k/2
                        img[i][j][2,3]=img[i][j][2,3]-k/2
            if mark[i][j]==0:#如果水印对应像素点为黑，则设置(2,2)位置像素值小于(2,3)
                if img[i][j][2,2]>img[i][j][2,3]:
                    t=img[i][j][2,2]
                    img[i][j][2,2]=img[i][j][2,3]
                    img[i][j][2,3]=t
                    if img[i][j][2,3]-img[i][j][2,2]<k:
                        img[i][j][2,2]=img[i][j][2,2]-k/2
                        img[i][j][2,3]=img[i][j][2,3]+k/2
                        
def divide(img,r,c):#将图像像素矩阵划分为r*c的像素块，返回每一小块经过1级haar小波变换和2级haar小波变换后的结果
    for i in range(64):
        for j in range(64):
            tmp1_img = img[(i*r):(i*r+r), (j*c):(j*c)+c]#划分图像为8*8小块
            div_img_haar1[i][j] = haar(tmp1_img)#对每一小块做haar变换
            tmp2_img = div_img_haar1[i][j][0:4, 0:4]
            div_img_haar2[i][j] = haar(tmp2_img)#对变换后的低频区A继续做2级haar变换，并储存
    return div_img_haar1,div_img_haar2

#置乱水印
p = arnold(watermark)
n = 20
for i in range(n-1):   
    p = arnold(p)

#乱水印单通道处理（原先有4通道）
mes_mark = np.zeros((64,64))
for i in range(64):
    for j in range(64):
        if p[i][j][0]==255:
            mes_mark[i][j]=255#得到单通道水印置乱灰度图

#随机二次置乱
key = []
for i in range(64):
    for j in range(64):
        rand = random.randint(1,100)
        if rand%2==1:
            mes_mark[i][j]=255-mes_mark[i][j]
        key.append(rand)
file = open('figs/key.txt','w')
file.write(str(key))

#图像像素块划分（每块8*8），对每一小块进行haar小波变换，对其中的低频区再做一次小波变换
div_img_haar1 = np.zeros((64,64,8,8))#用于储存图片划分后的8*8图像块
div_img_haar2 = np.zeros((64,64,4,4))
row_step = 8 #划分区块的大小
col_step = 8
div_img_haar1,div_img_haar2 = divide(img,row_step,col_step)

#注入水印        
enmark(div_img_haar2,mes_mark)

#注入水印后，使用haar逆变换，将2级haar处理结果还原为像素矩阵
back_img_haar1 = np.zeros((64,64,4,4))#haar2级到1级中间储存量
back_img = np.zeros((64,64,8,8))#处理后的图像矩阵，不过这时仍是被分开的状态
for i in range(64):
    for j in range(64):
        A1 = np.array(div_img_haar2[i][j][0:2,0:2])
        H1 = np.array(div_img_haar2[i][j][0:2,2:4])
        V1 = np.array(div_img_haar2[i][j][2:4,0:2])
        D1 = np.array(div_img_haar2[i][j][2:4,2:4])
        back_img_haar1[i][j] = pywt.idwt2((A1,(H1,V1,D1)),'haar')#haar逆变换还原图像，此处为2级haar还原为1级haar
        div_img_haar1[i][j][0:4, 0:4] = pywt.idwt2((A1,(H1,V1,D1)),'haar')#将处理后的haar放入原1级haar的低频区域
        A2 = np.array(div_img_haar1[i][j][0:4,0:4])
        H2 = np.array(div_img_haar1[i][j][0:4,4:8])
        V2 = np.array(div_img_haar1[i][j][4:8,0:4])
        D2 = np.array(div_img_haar1[i][j][4:8,4:8])
        back_img[i][j] = pywt.idwt2((A2,(H2,V2,D2)),'haar')#1级haar还原到图像

#把结果矩阵拼成最终图像
temp_list = []#储存每一行的拼接结果
for i in range(64):
    temp = back_img[i][0]
    for j in range(63):
        temp = np.hstack((temp,back_img[i][j+1]))#把每一行的8*8像素矩阵水平组合
    temp_list.append(temp)
result = np.vstack(temp_list)#每一行水平组合后，再竖直组合

plt.figure('origin_img')
plt.xlabel('origin_img')
plt.imshow(img,cmap='gray')
plt.figure('watermark')
plt.xlabel('watermark')
plt.imshow(watermark,cmap='gray')
plt.figure('messed_watermark')
plt.xlabel('messed_watermark')
plt.imshow(mes_mark,cmap='gray')
cv2.imwrite("figs/messed_watermark.png", mes_mark)
plt.figure('result')
plt.xlabel('result')
plt.imshow(result,cmap='gray')
cv2.imwrite("figs/result.png", result)#注意，如果此处输出jpg则提取的水印噪声会比较大，而如果是png则会小很多
#jpg压缩会损失信息


####提取水印###

#水印图像划分预处理
img2 = cv2.imread('figs/result.png',cv2.IMREAD_GRAYSCALE)#获取待提取水印图片像素数组
div_img2_haar1 = np.zeros((64,64,8,8))
div_img2_haar2 = np.zeros((64,64,4,4))
div_img2_haar1,div_img2_haar2 = divide(img2,8,8)#分割为8*8独立像素块，haar2级处理

#根据水印注入规则反向推出水印像素
pre_mark = np.zeros((64,64))
for i in range(64):
    for j in range(64):
        if div_img2_haar2[i][j][2,2]>div_img2_haar2[i][j][2,3]:
            pre_mark[i][j] = 255#如果(2,2)位置像素值大于(2,3)，水印像素点对应像素值255
       
#使用密钥还原随机处理后的置乱水印
for i in range(64):
    for j in range(64):
        if key[i*64+j]%2==1:
            pre_mark[i][j]=255-pre_mark[i][j]

#对提取的乱水印继续进行arnold变换
p = arnold(pre_mark)
T = 48 #T为64*64 arnold矩阵周期
N = T-n #n为注入时置乱次数
for i in range(N-1):   #继续置乱了N次，总置乱T次，应回到初态
    p = arnold(p)
final_mark = np.zeros((64,64))
for i in range(64):
    for j in range(64):
        if p[i][j][0]==255:
            final_mark[i][j]=255 #单通道处理
            
plt.figure('pre_mark')
plt.xlabel('pre_mark')
plt.imshow(pre_mark,cmap='gray')
cv2.imwrite('figs/pickout_pre_watermark.png',pre_mark)
plt.figure('final_mark')
plt.xlabel('final_mark')
plt.imshow(final_mark,cmap='gray')
cv2.imwrite('figs/pickout_final_watermark.png',final_mark)


###算法评估###

PSNR = 10*(math.log10(512*512)+2*math.log10(max(img2.ravel()))-math.log10(abs(int(img.sum())-int(img2.sum()))))
print("PSNR:",PSNR)
NC = sum(sum(np.multiply(pre_mark,mes_mark)))/sum(sum(np.multiply(mes_mark,mes_mark)))
print("NC:",NC)

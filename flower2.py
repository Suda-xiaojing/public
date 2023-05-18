# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:17:03 2023

@author: hxiaojing
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

img_url = 'https://img.zcool.cn/community/0156cb59439764a8012193a324fdaa.gif'       # 背景图片的网址
st.markdown('''<style>.css-fg4pbf{background-image:url(''' + img_url + ''');
background-size:100% 100%;background-attachment:fixed;}</style>
''', unsafe_allow_html=True)                                                        # 修改背景样式
 
def trans(select):
    if select == '黑色':
        return 'black'
    elif select == '银色':
        return 'silver'
    elif select == '亮红色':
        return 'lightcoral'
    elif select == '棕色':
        return 'brown'
    elif select == '橙色':
        return 'orange'
    elif select == '金黄色':
        return 'gold'
    elif select == '黄色':
        return 'yellow'
    elif select == '绿色':
        return 'lawngreen'
    elif select == '天蓝色':
        return 'cyan'
    elif select == '紫色':
        return 'purple'
    elif select == '圆形':
        return 'o'
    elif select == '朝下三角':
        return 'v'
    elif select == '朝上三角形':
        return '^'
    elif select == '正方形':
        return 's'
    elif select == '五边形':
        return 'p'
    elif select == '星型':
        return '*'
    elif select == '六角形':
        return 'h'
    elif select == '+号':
        return '+'
    elif select == 'x号':
        return 'x'
    elif select == '小型菱形':
        return 'd'

#中间标题
st.title('K-Means交互式组件'. center(33,'-'))
#侧边栏
st.sidebar.expander("")
st.sidebar.subheader('在下方调节你的参数')

cluster_class=st.sidebar.selectbox('1.聚类数量：',list(range(2,10)))
minmaxscaler=st.sidebar.radio('2.是否归一化：',['是','否'])
figure=['花瓣长度','花瓣宽度','花萼长度','花萼宽度']
figure1=st.sidebar.selectbox('3.选择展示第一类特征：',['花瓣长度','花瓣宽度','花萼长度','花萼宽度'])
figure2=st.sidebar.selectbox('4.选择展示第二类特征：',['花瓣长度','花瓣宽度','花萼长度','花萼宽度'])

choice=pd.DataFrame()
for i in range(1,cluster_class+1):
    col1,col2=st.sidebar.columns(2)
    with col1:
        choice.loc[i-1, 'color']=trans( st.selectbox(f'第{i}类颜色',['黑色', '银色','亮红色','棕色','橙色','金黄色','黄色','天蓝色','紫色']))
    with col2:
        choice.loc[i-1,'shape']=trans(st.selectbox(f'第{i}类形状', ['圆形', '朝下三角', '朝上三角形', '正方形', '五边形', '星型', '六角形', '+号', 'x号', '小型菱形']))
#调用数据
iris=load_iris()
data=iris['data']

if minmaxscaler=='是':
    data=MinMaxScaler().fit_transform(data)
    
model=KMeans(n_clusters=cluster_class).fit(data)
data_done=np.c_[data,model.labels_]

fig,ax=plt.subplots()
for i in set(model.labels_):
    index=data_done[:,-1]==i
    color=choice.loc[i,'color']
    shape=choice.loc[i,'shape']
    x=data_done[index,figure.index(figure1)]
    y=data_done[index,figure.index(figure2)]
    ax.scatter(x,y,c=color,marker=shape)
    
font_dict=dict(fontsize=16,color='maroon',family='SimHei')
ax.set_xlabel(figure1,fontdict=font_dict)
ax.set_ylabel(figure2,fontdict=font_dict)
ax.set_title('散点图',fontdict=font_dict)
ax.legend(set(model.labels_))
ax.set_xticks([])
ax.set_yticks([])
st.pyplot(fig)

st.markdown('''<style>#root > div:nth-child(1) > div > div > div > div >
section.css-1lcbmhc.e1fqkh3o3 > div.css-1adrfps.e1fqkh3o2
{background:rgba(255,255,255,0.5)}</style>''', unsafe_allow_html=True)  

st.markdown('''<style>#root > div:nth-child(1) > div > div > div > div >
section.main.css-1v3fvcr.egzxvld3 > div > div > div
{background-size:100% 100% ;background:rgba(207,207,207,0.9);
color:red; border-radius:5px;} </style>''', unsafe_allow_html=True)   

 
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
from PIL import Image
import cv2

import torch

from streamlit_lottie import st_lottie

from colour import Color

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# from google_drive_downloader import GoogleDriveDownloader as gdd


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


# cloud_model_location = '1rTTqtAyP7AwGNNFMd-cGrx1A0mmO1XEY'
# https://drive.google.com/file/d/1rTTqtAyP7AwGNNFMd-cGrx1A0mmO1XEY/view?usp=sharing

# @st.cache(ttl=3600, max_entries=10)
# def load_model():
#     save_dest = Path('model')
#     save_dest.mkdir(exist_ok=True)
#     f_checkpoint = Path('model/best.pt')
#     if not f_checkpoint.exists():
#         with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
#             gdd.download_file_from_google_drive(file_id=cloud_model_location,
#                                     dest_path=f_checkpoint,
#                                     unzip=False)
#     detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=f_checkpoint, force_reload=True)
#     return detection_model

@st.cache(ttl=3600, max_entries=10)
def load_model():
    f_checkpoint = Path('models/best.pt')
    detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=f_checkpoint, force_reload=True)
    return detection_model

model = load_model()


# <------------------------------------------------------------------------->

# st.set_page_config(layout="wide")

st.title('Технологии ИИ в детской стоматологии')
    
# <------------------------------------------------------------------------->   

# lottie_streamlit = load_lottiefile("40503-gladiator-tooth.json")
# st_lottie(lottie_streamlit, speed=1, height=400, key="initial")


def get_detection(image):
    pred = model(image).pandas().xyxy[0].sort_values('confidence', ascending=False)
    pred = pred[(pred.name == 'teeth')&(pred.confidence > 0.1)].reset_index(drop=True)
    bboxes = [[y for y in x] for x in pred[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(int)]
    return bboxes

def get_segmentation(image):
    pred = model(image).pandas().xyxy[0].sort_values('confidence', ascending=False)
    pred = pred[(pred.name == 'caries')&(pred.confidence > 0.0)].reset_index(drop=True)
    poly = [[y for y in x] for x in pred[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(int)]
    mask = np.zeros((image.shape[0], image.shape[1]))
    for pol in poly:
        mask[pol[1]:pol[3], pol[0]:pol[2]] = 1
    conf = pred.confidence.mean()
    return mask, poly, conf

def process_image(image, bboxes, mask):
    processed_image = image.copy()
    for bbox in bboxes:
        x_min = bbox[0]
        x_max = bbox[2]
        y_min = bbox[1]
        y_max = bbox[3]
        processed_image = cv2.rectangle(processed_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)

    color = np.array([255, 0, 0], dtype='uint8')
    masked_img = np.where(mask[..., None], color, processed_image)
    processed_image = cv2.addWeighted(processed_image, 0.8, masked_img, 0.5, 0)
    
    return processed_image


st.header('Загрузка данных')
uploaded_files = st.file_uploader('Загрузите изображения (можно загружать несколько',
                                  accept_multiple_files=True)

@st.cache(ttl=3600)
def predict_all(uploaded_files):
    if uploaded_files: # if user uploaded file

        bboxes_all = []
        mask_all = []
        poly_all = []
        caries_conf_all = []

        with st.spinner('Создаются предсказания...'):
            for file in uploaded_files:
                image = np.array(Image.open(file))
                bboxes = get_detection(image) 
                mask, poly, caries_conf = get_segmentation(image)
                bboxes_all.append(bboxes)
                mask_all.append(mask)
                poly_all.append(poly)
                caries_conf_all.append(caries_conf)
        
        preds_done = True
        
        return preds_done, bboxes_all, mask_all, poly_all, caries_conf_all

if uploaded_files:
    preds_done, bboxes_all, mask_all, poly_all, caries_conf_all = predict_all(uploaded_files)


if uploaded_files and preds_done: # if user uploaded file

    options = st.multiselect(
     'Выберите изображения для просмотра результатов',
     [x.name for x in uploaded_files])

    
    for file, idx in zip([x for x in uploaded_files if x.name[:] in options], 
                    [i for i in range(len(uploaded_files)) if uploaded_files[i].name[:] in options]):
    
        image = np.array(Image.open(file))

        bboxes = bboxes_all[idx] 
        mask, poly, caries_conf = mask_all[idx], poly_all[idx], caries_conf_all[idx]

        processed_image = process_image(image, bboxes, mask)

        st.header(f'Результат для {file.name[:]}')

        # Распознанные участки
        # Верхняя и нижняя часть зубов
        # Количество зубов

        max_dist = 0
        for i in np.array(bboxes)[:, 1]:
            for j in np.array(bboxes)[:, 1]:
                if max(i, j) - min(i, j) > max_dist:
                    max_dist = max(i, j) - min(i, j)
        if max_dist > 70:
            X = [[x] for x in np.array(bboxes)[:, 1]]
            up_down_clusters = KMeans(n_clusters=2).fit_predict(X)
        else:
            X = [[x] for x in np.array(bboxes)[:, 1]]
            up_down_clusters = KMeans(n_clusters=1).fit_predict(X)
        
        if len(set(up_down_clusters)) == 1:
            col1, col2, col3 = st.columns([3, 3, 1])
            col1.subheader('Изначальное изображение')
            col1.image(image)
            col2.subheader('Обработанное изображение')
            col2.image(processed_image)
            col3.metric(label="Всего зубов", value=len(bboxes))
            st.text('На изображении присутствует только одна часть зубов (верх/низ)')
        
        else:
            col1, col2, col3 = st.columns([3, 3, 1])
            col1.subheader('Изначальное изображение')
            col1.image(image)
            col2.subheader('Обработанное изображение')
            col2.image(processed_image)
            col3.metric(label="Всего зубов", value=len(bboxes))

            mean1 = np.mean([np.array(bboxes)[:, 1][i] for i in range(len(np.array(bboxes)[:, 1])) if up_down_clusters[i] == 1])
            mean0 = np.mean([np.array(bboxes)[:, 1][i] for i in range(len(np.array(bboxes)[:, 1])) if up_down_clusters[i] == 0])
            
            if mean1 > mean0:
                col3.metric(label='Верхних', value=len([bboxes[i] for i in range(len(bboxes)) if up_down_clusters[i] == 1]))
                col3.metric(label='Нижних', value=len([bboxes[i] for i in range(len(bboxes)) if up_down_clusters[i] == 0]))
                upmean = mean0
                downmean = mean1

            if mean0 > mean1:
                col3.metric(label='Верхних', value=len([bboxes[i] for i in range(len(bboxes)) if up_down_clusters[i] == 0]))
                col3.metric(label='Нижних', value=len([bboxes[i] for i in range(len(bboxes)) if up_down_clusters[i] == 1]))
                upmean = mean1
                downmean = mean0

            st.text('На изображении присутствуют и верхняя и нижняя часть зубов')


        # Проблемные участки
        if len(poly) == 0:
            st.subheader('Кариеса не обнаружено')

        else:
            st.subheader('Кариес обнаружен')
            
            poly_area = 0
            for pol in poly:
                poly_area += (pol[2]-pol[0])*(pol[3]-pol[1])
            bbox_area = 0
            for bbox in bboxes:
                bbox_area += (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            caries_area = round(100*poly_area/bbox_area, 2)
            
            if caries_conf > 0.5 and len(poly) > 2 and caries_area > 5:
                st.text('Рекомендуем СРОЧНО обратиться к стоматологу')
            elif caries_conf > 0.1 and len(poly) > 0 and caries_area > 2:
                st.text('Рекомендуем обратиться к стоматологу')
            if caries_conf < 0.1 and len(poly) > 0 and caries_area > 0:
                st.text('Вероятность кариеса не высока')

            col1, col2, col3 = st.columns([1, 1, 1])
            col1.metric(label='Вероятность детекции кариеса', value=str(round(100*caries_conf))+'%')
            col2.metric(label='Количество воспаленных участков', value=str(len(poly)))
            col3.metric(label='Площадь воспаленных участков', value=str(caries_area)+'%')


            caries_image = image.copy()
            for pol in poly:
                caries_image = cv2.rectangle(caries_image, (pol[0], pol[1]), (pol[2], pol[3]), (255, 0, 0), 1)
            # color = np.array([255, 0, 0], dtype='uint8')
            # masked_img = np.where(mask[..., None], color, caries_image)
            # caries_image = cv2.addWeighted(caries_image, 0.8, masked_img, 0.5, 0)

            if len(set(up_down_clusters)) == 1:
                col1, col2 = st.columns([1, 1])
                col1.text('Воспаления в левой части')
                col2.text('Воспаления в правой части')
                for pol in poly:
                    if pol[0] < 256/2:
                        col1.image(caries_image[max(0, pol[1]-15):pol[3]+15, max(0, pol[0]-15):pol[2]+15, :], 
                                   width=200)
                    else:
                        col2.image(caries_image[max(0, pol[1]-15):pol[3]+15, max(0, pol[0]-15):pol[2]+15, :], 
                                   width=200)

            else:
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                col1.text('Лево-верх')
                col2.text('Лево-низ')
                col3.text('Право-верх')
                col4.text('Право-низ')
                for pol in poly:
                    if pol[0] < 256/2 and (pol[1] - upmean) < (pol[1] - downmean):
                        col1.image(caries_image[max(0, pol[1]-15):pol[3]+15, max(0, pol[0]-15):pol[2]+15, :], 
                                   width=200)
                    elif pol[0] < 256/2 and (pol[1] - upmean) > (pol[1] - downmean):
                        col2.image(caries_image[max(0, pol[1]-15):pol[3]+15, max(0, pol[0]-15):pol[2]+15, :], 
                                   width=200)
                    elif pol[0] > 256/2 and (pol[1] - upmean) < (pol[1] - downmean):
                        col3.image(caries_image[max(0, pol[1]-15):pol[3]+15, max(0, pol[0]-15):pol[2]+15, :], 
                                   width=200)
                    elif pol[0] > 256/2 and (pol[1] - upmean) > (pol[1] - downmean):
                        col4.image(caries_image[max(0, pol[1]-15):pol[3]+15, max(0, pol[0]-15):pol[2]+15, :], 
                                   width=200)

                        

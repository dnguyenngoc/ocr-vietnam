import os
os.chdir('./')
import time
import datetime
from settings import config
import io
from ocr_model import CompletedModel
import numpy as np
from helpers import image_utils
from PIL import Image
import cv2
import requests
import schedule
import translate


print('load model ...')
model = CompletedModel()
print('model is loaded.')


def job():
    res = requests.get('http://{host}:{port}/api/v1/ml-ocr/not-process?limit=50'.format(host=config.BE_HOST, port = config.BE_PORT))
    print('[Load] load 50 field', res.status_code)
    if res.status_code != 200:
        print(res.json())
        return None
    imports= res.json()
    document_id_check = imports[0]['document_id']
    print(document_id_check)
    for item in imports:
        if item['document_id'] != document_id_check:
            data = {'document_id': document_id_check, 'status_code': 300}
            res = requests.put('http://{host}:{port}/api/v1/ml-ocr/doc-export'.format(host=config.BE_HOST, port = config.BE_PORT), data = data)
            print(res.status_code)
            document_id_check = item['document_id']
            print('[RUN-DOC]: ', item['document_id'])
        
        if item['url'] == None:
            continue
        print("[RUN-FIELD] ", "id: ", item['id'], 'field_id: ', item['field_id'], 'document_id: ', item['document_id'])
        image = image_utils.read_image_from_url(item['url'])
        image = np.asarray(image)
        text = model.predict(image)
        print(text)
        data = {'id': item['id'], 'document_id': item['document_id'], 'value': text}
        res = requests.put('http://{host}:{port}/api/v1/ml-ocr/complete-process'.format(host=config.BE_HOST, port = config.BE_PORT), data =data)
    data = {'document_id': item['document_id'], 'status_code': 300}
    res = requests.put('http://{host}:{port}/api/v1/ml-ocr/doc-export'.format(host=config.BE_HOST, port = config.BE_PORT), data = data)
    print(res.status_code)
    document_id_check = item['document_id']
    print('[RUN-DOC]: ', item['document_id'])
        
    print("[COMPLETED]")

schedule.every(1).seconds.do(job)
while True:
    schedule.run_pending()
    time.sleep(10)

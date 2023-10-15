#! .\venv\
# encoding: utf-8
# @Time   : 2023/7/3
# @Author : Spike
# @Descr   :
import os.path
import requests
from comm_tools import func_box
from paddleocr import PaddleOCR, draw_ocr
import concurrent.futures
from comm_tools import database_processor
# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`


class Paddle_ocr_select():

    def __init__(self, ipaddr='', trust_value=0.9):
        self.font_path = os.path.join(func_box.base_path, 'docs/OCR/fonts/simfang.ttf')
        self.ipaddr = ipaddr
        self.trust_value = trust_value

    def img_def_content(self, img_path, img_tag, show_result: bool=True):
        model_dir = os.path.join(func_box.base_path, 'docs', 'OCR', 'ch_PP-OCRv3_rec_infer')
        det_dir = os.path.join(func_box.base_path, 'docs', 'OCR', 'ch_PP-OCRv3_det_infer')
        cls_dir = os.path.join(func_box.base_path, 'docs', 'OCR', 'ch_ppocr_mobile_v2.0_cls_infer')
        ocr = PaddleOCR(use_angle_cls=True, cls_model_dir=cls_dir,
                        rec_model_dir=model_dir, det_model_dir=det_dir)
        save_path = os.path.join(func_box.users_path, self.ipaddr, 'ocr_temp')
        os.makedirs(save_path, exist_ok=True)
        if img_path.startswith('http'):
            response = requests.get(url=img_path, verify=False)
            img_path = os.path.join(save_path, f'{img_tag}.jpeg')
            with open(img_path, mode='wb') as f: f.write(response.content)
        result = ocr.ocr(img_path, cls=True)
        save_file = os.path.join(save_path, f'{img_tag}.jpeg')
        result = result[0]
        draw_error = False
        if show_result:
            # 显示结果
            from PIL import Image
            boxes = [line[0] for line in result if line[1][1] > self.trust_value]
            txts = [line[1][0] for line in result]
            txts_select = [line[1][0] for line in result if line[1][1] > self.trust_value]
            scores = [line[1][1] for line in result]
            try:
                image = Image.open(img_path).convert('RGB')
                im_show = draw_ocr(image, boxes, txts, scores, font_path=self.font_path)
                im_show = Image.fromarray(im_show)
                im_show.save(save_file)
            except Exception:
                print('绘制选择文字出错')
                save_file = img_path
                draw_error = '右侧无文案说明仅代表绘制选择文字, 不影响实际OCR成果'
        else:
            save_file = ''
            txts_select = result
        return '\n'.join(txts_select), save_file, draw_error

    def identify_cache(self, img_path, img_tag, show_result: bool=True):
        cache_sql = database_processor.SqliteHandle('ocr_cache')
        cache = cache_sql.get_prompt_value(find=img_tag)
        temp_cont, file_path = cache
        if file_path:
            content = temp_cont[img_tag]
            error = '本次识别结果读取数据库缓存'
        else:
            content, file_path, error = self.img_def_content(img_path, img_tag, show_result)
            cache_sql.inset_prompt({img_tag: content}, file_path)
        return content, file_path, error


def submit_threads_ocr(dictionary, func, max_threads=10):
    threads = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_threads)
    # 提交任务，并将线程对象作为键，字典的键作为值存储
    for key in dictionary:
        threads[key] = executor.submit(func, dictionary[key], key)
    # 返回线程字典
    return threads


if __name__ == '__main__':
    # 测试
    my_dict = {'key1': '/Users/kilig/Job/Python-project/kso_gpt/private_upload/img.png',
               'key2': '/Users/kilig/Job/Python-project/kso_gpt/private_upload/img.png'}
    thread_dict = submit_threads_ocr(my_dict, Paddle_ocr_select().identify_cache, 2)
    for t in thread_dict:
        print('1232312')
        print(thread_dict[t].result())

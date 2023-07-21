#! .\venv\
# encoding: utf-8
# @Time   : 2023/6/14
# @Author : Spike
# @Descr   :
import re, os
import json, time
from bs4 import BeautifulSoup
import requests
import sys
job_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(job_path)
from comm_tools import func_box, ocr_tools, toolbox
from openpyxl import load_workbook
import urllib.parse
import gradio as gr



class Utils:

    def __init__(self):
        self.find_keys_type = 'type'
        self.find_picture_source = ['caption', 'imgID', 'sourceKey']
        self.find_document_source = ['wpsDocumentLink', 'wpsDocumentName', 'wpsDocumentType']
        self.find_document_tags = ['WPSDocument']
        self.find_picture_tags = ['picture', 'processon']
        self.picture_format = ['.JPEG', '.PNG', '.GIF', '.BMP', '.TIFF']

    def find_all_text_keys(self, dictionary, parent_type=None, text_values=None, filter_type=''):
        """
        Args:
            dictionary: 字典或列表
            parent_type: 匹配的type，作为新列表的key，用于分类
            text_values: 存储列表
            filter_type: 当前层级find_keys_type==filter_type时，不继续往下嵌套
        Returns:
            text_values和排序后的context_
        """
        # 初始化 text_values 为空列表，用于存储找到的所有text值
        if text_values is None:
            text_values = []
        # 如果输入的dictionary不是字典类型，返回已收集到的text值
        if not isinstance(dictionary, dict):
            return text_values
        # 获取当前层级的 type 值
        current_type = dictionary.get(self.find_keys_type, parent_type)
        # 如果字典中包含 'text' 或 'caption' 键，将对应的值添加到 text_values 列表中
        if 'text' in dictionary:
            content_value = dictionary.get('text', None)
            text_values.append({current_type: content_value})
        if 'caption' in dictionary:
            temp = {}
            for key in self.find_picture_source:
                temp[key] = dictionary.get(key, None)
            text_values.append({current_type: temp})
        if 'wpsDocumentId' in dictionary:
            temp = {}
            for key in self.find_document_source:
                temp[key] = dictionary.get(key, None)
            text_values.append({current_type: temp})
        # 如果当前类型不等于 filter_type，则继续遍历子级属性
        if current_type != filter_type:
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    self.find_all_text_keys(value, current_type, text_values, filter_type)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            self.find_all_text_keys(item, current_type, text_values, filter_type)
        return text_values

    def statistical_results(self, text_values, img_proce=False):
        context_ = []
        pic_dict = {}
        file_dict = {}
        for i in text_values:
            for key, value in i.items():
                if key in self.find_picture_tags:
                    if img_proce:
                        mark = f'{key}OCR结果: """{value["sourceKey"]}"""\n'
                        if value["caption"]: mark += f'{key}描述: {value["caption"]}\n'
                        context_.append(mark)
                        pic_dict[value['sourceKey']] = value['imgID']
                    else:
                        if value["caption"]: context_.append(f'{key}描述: {value["caption"]}\n')
                        pic_dict[value['sourceKey']] = value['imgID']
                elif key in self.find_document_tags:
                    mark = f'{value["wpsDocumentName"]}: {value["wpsDocumentLink"]}'
                    context_.append(mark)
                else:
                    context_.append(value)
        context_ = '\n'.join(context_)
        return text_values, context_, pic_dict, file_dict

    def write_markdown(self, data, hosts, file_name):
        user_path = os.path.join(func_box.users_path, hosts, 'markdown')
        os.makedirs(user_path, exist_ok=True)
        md_file = os.path.join(user_path, f"{file_name}.md")
        with open(file=md_file, mode='w') as f:
            f.write(data)
        return md_file

    def markdown_to_flow_chart(self, data, hosts, file_name):
        user_path = os.path.join(func_box.users_path, hosts, 'mark_map')
        md_file = self.write_markdown(data, hosts, file_name)
        html_file = os.path.join(user_path, f"{file_name}.html")
        func_box.Shell(f'npx markmap-cli --no-open "{md_file}" -o "{html_file}"').read()
        return md_file, html_file

    def split_startswith_txt(self, link_limit, start='http'):
        link = str(link_limit).split()
        links = []
        for i in link:
            if i.startswith(start):
                links.append(i)
        return links

    def global_search_for_files(self, file_path, matching: list):
        file_list = []
        for root, dirs, files in os.walk(file_path):
            for file in files:
                for math in matching:
                    if str(math).lower() in str(file).lower():
                        file_list.append(os.path.join(root, file))
        return file_list



class ExcelHandle:

    def __init__(self, ipaddr, is_client=True):
        if type(is_client) is bool and is_client:
            self.template_excel = os.path.join(func_box.base_path, 'docs/template/【Temp】测试要点.xlsx')
        elif not is_client:
            self.template_excel = os.path.join(func_box.base_path, 'docs/template/接口测试用例模板.xlsx')
        elif type(type) is str:
            if os.path.exists(is_client):
                self.template_excel = is_client
            else:
                self.template_excel = os.path.join(func_box.base_path, 'docs/template/【Temp】测试要点.xlsx')
        self.user_path = os.path.join(func_box.base_path, 'private_upload', ipaddr, 'test_case')
        os.makedirs(f'{self.user_path}', exist_ok=True)

    def lpvoid_lpbuffe(self, data_list: list, filename='', decs=''):
        # 加载现有的 Excel 文件
        workbook = load_workbook(self.template_excel)
        # 选择要操作的工作表
        worksheet = workbook['测试要点']
        try:
            decs_sheet = workbook['说明']
            decs_sheet['C2'] = decs
        except:
            print('文档没有说明的sheet')
        # 定义起始行号
        start_row = 4
        # 遍历数据列表
        for row_data in data_list:
            # 写入每一行的数据到指定的单元格范围
            for col_num, value in enumerate(row_data, start=1):
                cell = worksheet.cell(row=start_row, column=col_num)
                cell.value = value
            # 增加起始行号
            start_row += 1
        # 保存 Excel 文件
        time_stamp = time.strftime("%Y-%m-%d-%H", time.localtime())
        if filename == '': filename = time.strftime("%Y-%m-%d-%H", time.localtime()) + '_temp'
        else: f"{time_stamp}_{filename}"
        test_case_path = f'{os.path.join(self.user_path, filename)}.xlsx'
        workbook.save(test_case_path)
        return test_case_path


class Kdocs:

    def __init__(self, url):
        WPS_COOKIES, = toolbox.get_conf('WPS_COOKIES',)
        self.url = url
        self.cookies = WPS_COOKIES
        self.headers = {
            'accept-language': 'en-US,en;q=0.9,ja;q=0.8',
            'content-type': 'text/plain;charset=UTF-8',
            'x-csrf-rand': '',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7'}
        self.ex_headers = {
            'Host': 'www.kdocs.cn',
            'accept': 'application/json, text/plain, */*',
            'content-type': 'application/json',
            'sec-ch-ua-platform': '"macOS"',
            'origin': 'https://www.kdocs.cn',
            'referer': 'https://www.kdocs.cn/l/cizMZqbz9FiZ?R=L1MvNA==',
        }
        self.parm_otl_data = {"connid": "",
                              "args": {"password": "", "readonly": False, "modifyPassword": "", "sync": True,
                                       "startVersion": 0, "endVersion": 0},
                              "ex_args": {"queryInitArgs": {"enableCopyComments": False, "checkAuditRule": False}},
                              "group": "", "front_ver": ""}
        self.parm_shapes_data = {"objects": [], "expire": 86400000, "support_webp": True, "with_thumbnail": True,
                                 "support_lossless": True}
        self.parm_export_preload = {"ver": "56",
                                     "csrfmiddlewaretoken": "PHMp62MZYXzzAPnH2drHay7NW2sHrw8c"}
        self.tol_url = 'https://www.kdocs.cn/api/v3/office/file/%v/open/otl'
        self.shapes_url = 'https://www.kdocs.cn/api/v3/office/file/%v/attachment/shapes'
        self.kdocs_download_url = 'https://drive.kdocs.cn/api/v5/groups/%g/files/%f/download?isblocks=false&support_checksums=md5,sha1,sha224,sha256,sha384,sha512'
        self.drive_download_url = 'https://drive.wps.cn/api/v3/groups/%g/files/%f/download?isblocks=false'
        self.group_url = 'https://drive.wps.cn/api/v5/links/%v?review=true'
        self.export_url = 'https://www.kdocs.cn/api/v3/office/file/%f/export/%t/result'
        self.preload_url = 'https://www.kdocs.cn/api/v3/office/file/%f/export/%t/preload'
        self.url_share_tag = self.split_link_tags()
        self.file_info_parm = self.get_file_info_parm()
        self.docs_old_type = ['docs', 'doc', 'pptx', 'ppt', 'xls', 'xlsx', 'pdf', 'csv', 'txt']
        self.media_type = ['.mp4', '.m4a', '.wav', '.mpga', '.mpeg', '.mp3', '.avi', '.mkv', '.flac', '.aac']
        self.smart_type = {'.otl': 'pdf', '.ksheet': 'xlsx'}

    def get_file_info_html(self):
        """
        获取传递过来的文档HTML信息
        Returns:
            HTML信息
        """
        response = requests.get(self.url, cookies=self.cookies, headers=self.headers)
        return response.text

    def get_file_info_parm(self):
        response = requests.get(self.group_url.replace("%v", self.url_share_tag),
                                cookies=self.cookies,
                                headers=self.headers, verify=False).json()
        file_info = response['fileinfo']
        return file_info

    def document_aggregation_download(self, file_type=''):
        link_name = self.file_info_parm['fname']
        link = ''
        for t in self.docs_old_type:
            if t in link_name and file_type in link_name:
                link = self.get_docs_old_link()
        for t in self.media_type:
            if t in link_name and file_type in link_name:
                link = self.get_media_link()
        for t in self.smart_type:
            if t in link_name and file_type in link_name:
                link = self.get_kdocs_intelligence_link(type=self.smart_type[t])
                link_name = link_name+f".{self.smart_type[t]}"
        return link, link_name

    def get_media_link(self):
        response = requests.get(self.drive_download_url.replace("%g", str(self.file_info_parm['groupid'])
                                                                ).replace('%f', str(self.file_info_parm['id'])),
                                cookies=self.cookies,
                                headers=self.headers, verify=False)
        link = response.json()['fileinfo']['url']
        return self.url_decode(link)

    def get_docs_old_link(self):
        response = requests.get(self.kdocs_download_url.replace("%g", str(self.file_info_parm['groupid'])
                                                                ).replace('%f', str(self.file_info_parm['id'])),
                                cookies=self.cookies,
                                headers=self.headers, verify=False)
        try:
            link = response.json()['fileinfo']['url']
        except:
            link = response.json()['url']
        return self.url_decode(link)

    def get_kdocs_intelligence_link(self, type='xlsx'):
        response_task = requests.post(
            self.preload_url.replace('%f', str(self.file_info_parm['id'])).replace('%t', type),
            cookies=self.cookies,
            headers=self.ex_headers,
            json=self.parm_export_preload, verify=False
        )
        self.parm_export_preload.update(response_task.json())
        for i in range(20):
            response_link = requests.post(
                self.export_url.replace('%f', str(self.file_info_parm['id'])).replace('%t', type),
                cookies=self.cookies,
                headers=self.ex_headers,
                json=self.parm_export_preload, verify=False
            )
            if response_link.json()['status'] == 'finished':
                return response_link.json()['data']['url']
        return None

    def download_share_file(self):
        response = requests.get(self.preload_url.replace("%g", str(self.file_info_parm['groupid'])
                                                                ).replace('%f', str(self.file_info_parm['id'])),
                                cookies=self.cookies,
                                headers=self.headers, verify=False)

    def split_link_tags(self):
        url_parts = re.split('[/\?&#]+', self.url)
        try:
            l_index = url_parts.index('l')
            otl_url_str = url_parts[l_index + 1]
            return otl_url_str
        except ValueError:
            return None

    def get_file_content(self):
        """
        爬虫解析文档内容
        Returns:
            文档内容
        """
        otl_url_str = self.split_link_tags()
        if otl_url_str is None: return
        html_content = self.get_file_info_html()
        self.bs4_file_info(html_content)  # 调用 bs4_file_info() 方法解析 html_content，获取文件信息# 更新类的parm_data 和 headers
        json_data = json.dumps(self.parm_otl_data)
        response = requests.post(
            str(self.tol_url).replace('%v', otl_url_str),
            cookies=self.cookies,
            headers=self.headers,
            data=json_data,)
        return response.json(), response.text

    def get_file_pic_url(self, pic_dict: dict):
        otl_url_str = self.split_link_tags()
        if otl_url_str is None: return
        for pic in pic_dict:
            pic_parm = {'attachment_id': pic, "imgId": pic_dict[pic], "max_edge": 1180, "source": ""}
            self.parm_shapes_data['objects'].append(pic_parm)
        json_data = json.dumps(self.parm_shapes_data)
        response = requests.post(
            str(self.shapes_url).replace('%v', otl_url_str),
            cookies=self.cookies,
            headers=self.headers,
            data=json_data,)
        url_data = response.json()['data']
        for pic in url_data:
            try:
                pic_dict[pic] = self.url_decode(url_data[pic]['url'])
            except Exception as f:
                pass
        return pic_dict

    @staticmethod
    def url_decode(url):
        decoded_url = urllib.parse.unquote(url)
        return decoded_url


    def bs4_file_info(self, html_str):
        """
        bs4爬虫文档信息，没有这个可不行🤨
        Args:
            html_str: HTML信息
        Returns:
            {'connid': 文档id, 'group': 文档的群组, 'front_ver': 文档版本}
        """
        html = BeautifulSoup(html_str, "html.parser")
        # Find all script tags in the HTML
        script_tags = html.find_all("script")
        json_string = None
        # Iterate through script tags to find the one containing required data
        for tag in script_tags:
            if tag.string and "window.__WPSENV__" in tag.string:
                json_string = re.search(r"window\.__WPSENV__=(.*);", tag.string).group(1)
                break
        if json_string:
            # Load the JSON data from the found string
            json_data = json.loads(json_string)
            file_connid = json_data['conn_id']
            file_group = json_data['user_group']
            file_front_ver = json_data['file_version']
            file_id = json_data['root_file_id']
            group_id = json_data['file_info']['file']['group_id']
            self.headers['x-csrf-rand'] = json_data['csrf_token']
            self.parm_otl_data.update({'connid': file_connid, 'group': file_group, 'front_ver': file_front_ver,
                                       'file_id': file_id, 'group_id':group_id})
            return True
        else:
            return None

def get_docs_content(url, image_processing=False):
    kdocs = Kdocs(url)
    utils = Utils()
    json_data, json_dict = kdocs.get_file_content()
    text_values = utils.find_all_text_keys(json_data, filter_type='')
    _all, content, pic_dict, file_dict = utils.statistical_results(text_values, img_proce=image_processing)
    pic_dict_convert = kdocs.get_file_pic_url(pic_dict)
    empty_picture_count = sum(1 for item in _all if 'picture' in item and not item['picture']['caption'])
    return _all, content, empty_picture_count, pic_dict_convert


def get_kdocs_from_everything(txt, type='', ipaddr='temp'):
    """
    Args:
        txt: kudos 文件分享码
        type: type=='' 时，将支持所有文件类型
        ipaddr:

    Returns:
    """
    link_limit = Utils().split_startswith_txt(link_limit=txt)
    file_manifest = []
    success = False
    project_folder = os.path.join(func_box.users_path, ipaddr, 'kdocs')
    if link_limit:
        for limit in link_limit:
            name = 'temp.txt'
            os.makedirs(project_folder, exist_ok=True)
            if type == 'otl':
                _, content, _, pic_dict = get_docs_content(limit)
                for i in pic_dict:  # 增加OCR选项
                    img_content, img_result = ocr_tools.Paddle_ocr_select(ipaddr=ipaddr, trust_value=True
                                                                          ).img_def_content(img_path=pic_dict[i])
                    content = str(content).replace(f"{i}", f"{func_box.html_local_img(img_result)}\n```{img_content}```")
                    name = content.splitlines()[0] + '.md'
                    content = content.encode('utf-8')
            elif type or type == '':
                link, name = Kdocs(limit).document_aggregation_download(file_type=type)
                if link:
                    resp = requests.get(url=link, verify=False)
                    content = resp.content
                else:
                    return False, [], []
            else:
                return False, [], []
            if content:
                temp_file = os.path.join(project_folder, name)
                with open(temp_file, 'wb') as f: f.write(content)
                file_manifest.append(temp_file)
                success = True
    return success, file_manifest, project_folder


def json_args_return(kwargs, keys: list) -> list: 
    temp = [False for i in range(len(keys))]
    for i in range(len(keys)):
        try:
            temp[i] = json.loads(kwargs['advanced_arg'])[keys[i]]
        except Exception as f:
            try:
                temp[i] = kwargs['parameters_def'][keys[i]]
            except Exception as f:
                temp[i] = False
    return temp


def ocr_batch_processing(file_manifest, chatbot, history, llm_kwargs):
    ocr_process = f'> 红框为采用的文案,可信度低于 {func_box.html_tag_color(llm_kwargs["ocr"])} 将不采用, 可在Setting 中进行配置\n\n'
    i_say = 'ORC开始工作'
    chatbot.append([i_say, ocr_process])
    for pic_path in file_manifest:
        yield from toolbox.update_ui(chatbot, history, '正在调用OCR组件，图片多可能会比较慢')
        img_content, img_result = ocr_tools.Paddle_ocr_select(ipaddr=llm_kwargs['ipaddr'],
                                                              trust_value=llm_kwargs['ocr']
                                                              ).img_def_content(img_path=pic_path)
        ocr_process += f'{pic_path} 识别完成，识别效果如下 {func_box.html_local_img(img_result)} \n\n' \
                       f'```\n{img_content}\n```'
        chatbot[-1] = [i_say, ocr_process]
        yield from toolbox.update_ui(chatbot, history)
    ocr_process += f'\n\n---\n\n解析成功，现在我已理解上述内容，有什么不懂得地方你可以问我～'
    chatbot[-1] = [i_say, ocr_process]
    history.extend([i_say, ocr_process])
    yield from toolbox.update_ui(chatbot, history)


def ocr_batch_plugin(txt, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, web_port):
    chatbot_with_cookie = toolbox.ChatBotWithCookies(chatbot)
    chatbot_with_cookie.write_list(chatbot)
    file_handle = Utils()
    file_manifest = file_handle.split_startswith_txt(txt, start='http')
    correction_copy = f'如果是本地文件，请点击【UPLOAD】先上传，多个文件请上传压缩包，'\
                      f'如果是网络文件，请粘贴到输入框，'\
                      f'多个文件{func_box.html_tag_color("请使用换行或空格区分")}'
    if txt:
        if os.path.exists(txt):
            file_manifest = file_handle.global_search_for_files(txt, matching=file_handle.picture_format)
            yield from ocr_batch_processing(file_manifest, chatbot, history, llm_kwargs=llm_kwargs)
        elif file_manifest != []:
            yield from ocr_batch_processing(file_manifest, chatbot, history, llm_kwargs=llm_kwargs)
        else:
            chatbot.append([correction_copy, None])
            yield from toolbox.update_ui(chatbot, history)
    else:
        chatbot.append([f'空空如也的输入框，{correction_copy}', None])
        yield from toolbox.update_ui(chatbot, history)


if __name__ == '__main__':
    import time
    print(get_kdocs_from_everything('https://www.kdocs.cn/l/cgkva5MzXmey'))
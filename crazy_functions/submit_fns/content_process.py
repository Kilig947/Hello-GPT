# encoding: utf-8
# @Time   : 2024/7/19
# @Author : Spike
# @Descr   :
import os
import json
import re
import copy
from common.logger_handler import logger
from common import gr_converter_html
from common.toolbox import update_ui,  get_conf, trimmed_format_exc
from common.path_handler import init_path
from crazy_functions import reader_fns
from common.func_box import (split_domain_url, extract_link_pf, replace_special_chars,
                             vain_open_extensions, valid_img_extensions)


def find_index_inlist(data_list: list, search_terms: list) -> int:
    """ 在data_list找到符合search_terms字符串，找到后直接返回下标
    Args:
        data_list: list数据，最多往里面找两层
        search_terms: list数据，符合一个就返回数据
    Returns: 对应的下标
    """
    for i, sublist in enumerate(data_list):
        if any(term in str(sublist) for term in search_terms):
            return i
        for j, item in enumerate(sublist):
            if any(term in str(item) for term in search_terms):
                return i
    return 0  # 如果没有找到匹配的元素，则返回初始坐标


def file_reader_content(file_path, save_path, plugin_kwargs):
    reader_statsu = ''
    file_content = ''

    if file_path.endswith('pdf'):
        content = reader_fns.PDFHandler(file_path, save_path).get_markdown()
        file_content = "".join(content)
    elif file_path.endswith('docx') or file_path.endswith('doc'):
        file_content = reader_fns.DocxHandler(file_path, save_path).get_markdown()
    elif file_path.endswith('xmind'):
        file_content = reader_fns.XmindHandle(file_path, save_path).get_markdown()
    elif file_path.endswith('mp4'):
        file_content = reader_fns.AudioHandler(file_path).video_converters()
    elif file_path.endswith('xlsx') or file_path.endswith('xls'):
        sheet, = json_args_return(plugin_kwargs, keys=['读取指定Sheet'], default='测试要点')
        # 创建文件对象
        ex_handle = reader_fns.XlsxHandler(file_path, save_path, sheet=sheet)
        if sheet in ex_handle.workbook.sheetnames:
            ex_handle.split_merged_cells(save_work=False)  # 避免更改到原文件
            xlsx_dict = ex_handle.read_as_dict()
            file_content = xlsx_dict.get(sheet)
        else:
            active_sheet = ex_handle.workbook.active.title
            ex_handle.sheet = active_sheet
            ex_handle.split_merged_cells(save_work=False)
            xlsx_dict = ex_handle.read_as_dict()
            file_content = xlsx_dict.get(active_sheet)
            reader_statsu += f'\n\n无法在`{os.path.basename(file_path)}`找到`{sheet}`工作表，' \
                             f'将读取上次预览的活动工作表`{active_sheet}`，' \
                             f'若你的用例工作表是其他名称, 请及时暂停插件运行，并在自定义插件配置中更改' \
                             f'{gr_converter_html.html_tag_color("读取指定Sheet")}。'
        plugin_kwargs['写入指定模版'] = file_path
        plugin_kwargs['写入指定Sheet'] = ex_handle.sheet
    elif file_path.split('.')[-1] not in vain_open_extensions:
        try:
            with open(file_path, mode='r', encoding='utf-8') as f:
                file_content = f.read()
        except Exception as e:
            reader_statsu += f'An error occurred while reading the file: {e}'
    return file_content, reader_statsu


def file_extraction_intype(file_mapping, chatbot, history, llm_kwargs, plugin_kwargs):
    # 文件读取
    file_limit = {}
    file_format = gr_converter_html.get_fold_panel()
    old_say = chatbot[-1][1] + '\n\n'
    for file_path in file_mapping:
        chatbot[-1][1] = old_say + file_format(
            title=f'正在解析本地文件:【{file_path.replace(init_path.base_path, ".")}】')
        yield from update_ui(chatbot, history)
        save_path = os.path.join(init_path.private_files_path, llm_kwargs['ipaddr'])
        content, status = file_reader_content(file_path, save_path, plugin_kwargs)
        if isinstance(content, str):
            file_limit[file_path] = content.replace(init_path.base_path, 'file=.')
        else:
            file_limit[file_path] = content
        mapping_data = "\n\n--\n\n".join([f"{file_limit[fp]}" for fp in file_limit])
        chatbot[-1][1] = old_say + file_format(title=f'文件解析完成', content=mapping_data, status='Done')
        yield from update_ui(chatbot, history, msg=status)
    return file_limit


def json_args_return(kwargs: dict, keys: list, default=None) -> list:
    """
    Args: 提取插件的调优参数，如果有，则返回取到的值，如果无，则返回False
        kwargs: 一般是plugin_kwargs
        keys: 需要取得key
        default: 找不到时总得返回什么东西
    Returns: 有key返value，无key返False
    """
    temp = [default for i in range(len(keys))]
    for i in range(len(keys)):
        try:
            temp[i] = kwargs[keys[i]]
        except Exception:
            try:
                temp[i] = json.loads(kwargs['advanced_arg'])[keys[i]]
            except Exception as f:
                try:
                    temp[i] = kwargs['parameters_def'][keys[i]]
                except Exception as f:
                    temp[i] = default
    return temp


def check_url_domain_cloud(link_limit):
    wps_links = split_domain_url(link_limit, domain_name=[get_conf('WPS_BASE_HOST'), 'wps'])
    qq_link = split_domain_url(link_limit, domain_name=[get_conf('QQ_BASE_HOST')])
    feishu_link = split_domain_url(link_limit, domain_name=[get_conf('FEISHU_BASE_HOST')])
    project_link = split_domain_url(link_limit, domain_name=[get_conf('PROJECT_BASE_HOST')])
    return wps_links, qq_link, feishu_link, project_link


def detach_cloud_links(link_limit, llm_kwargs, valid_types):
    fp_mapping = {}
    save_path = os.path.join(init_path.private_files_path, llm_kwargs['ipaddr'])
    wps_links, qq_link, feishu_link, project_link = check_url_domain_cloud(link_limit)
    wps_status, qq_status, feishu_status = '', '', ''
    try:
        # wps云文档下载
        wps_status, wps_mapping = reader_fns.get_kdocs_from_limit(wps_links, save_path, llm_kwargs.get('wps_cookies'))
        fp_mapping.update(wps_mapping)
    except Exception as e:
        error = trimmed_format_exc()
        wps_status += f'# 下载WPS文档出错了 \n ERROR: {error} \n'

    try:
        # qq云文档下载
        qq_status, qq_mapping = reader_fns.get_qqdocs_from_limit(qq_link, save_path, llm_kwargs.get('qq_cookies'))
        fp_mapping.update(qq_mapping)
    except Exception as e:
        error = trimmed_format_exc()
        wps_status += f'# 下载QQ文档出错了 \n ERROR: {error}'

    try:
        # 飞书云文档下载
        feishu_status, feishu_mapping = reader_fns.get_feishu_from_limit(feishu_link, save_path,
                                                                         llm_kwargs.get('feishu_header'))
        fp_mapping.update(feishu_mapping)
    except Exception as e:
        error = trimmed_format_exc()
        wps_status += f'# 下载飞书文档出错了 \n ERROR: {error}'

    try:
        # 飞书项目转换
        feishu_status, feishu_mapping = reader_fns.get_project_from_limit(project_link, save_path,
                                                                          llm_kwargs.get('project_config'))
        fp_mapping.update(feishu_mapping)
    except Exception as e:
        error = trimmed_format_exc()
        wps_status += f'# 解析飞书项目出错了 \n ERROR: {error}'

    download_status = ''
    if wps_status or qq_status or feishu_status:
        download_status = "\n".join([wps_status, qq_status, feishu_status]).strip('\n')
    pop_list = []
    # 筛选文件
    for fp in fp_mapping:
        if fp.split('.')[-1] not in valid_types and valid_types != ['*']:
            download_status += '\n\n' + f'过滤掉了`{fp_mapping[fp]}`，因为不是插件能够接收处理的文件类型`{valid_types}`'
    for fpop in pop_list:
        fp_mapping.pop(fpop)  # 过滤不能处理的文件
    return fp_mapping, {'status': download_status}


def content_clear_links(user_input, clear_fp_map, clear_link_map):
    """清除文本中已处理的链接"""
    for link in clear_link_map:
        user_input = user_input.replace(link, '')
    for pf in clear_fp_map:
        user_input = user_input.replace(clear_fp_map[pf], '')
    return user_input


def input_retrieval_file(user_input, llm_kwargs, valid_types):
    # 网络链接
    fp_mapping, download_status = detach_cloud_links(user_input, llm_kwargs, valid_types)
    # 本地文件
    fp_mapping.update(extract_link_pf(user_input, valid_types))
    return fp_mapping, download_status


def content_img_vision_analyze(content: str, chatbot, history, llm_kwargs, plugin_kwargs):
    ocr_switch, = json_args_return(plugin_kwargs, ['开启OCR'])
    cor_cache = llm_kwargs.get('ocr_cache', False)
    img_mapping = extract_link_pf(content, valid_img_extensions)
    gpt_old_say = chatbot[-1][1]
    vision_format = gr_converter_html.get_fold_panel()
    # 如果开启了OCR，并且文中存在图片链接，处理图片
    if ocr_switch and img_mapping:
        vision_loading_statsu = {os.path.basename(i): "Loading..." for i in img_mapping}
        chatbot[-1][1] = gpt_old_say + vision_format(f'检测到识图开关为`{ocr_switch}`，正在识别图片中的文字...',
                                                     vision_loading_statsu)
        yield from update_ui(chatbot=chatbot, history=history)
        # 识别图片中的文字
        save_path = os.path.join(init_path.private_files_path, llm_kwargs['ipaddr'])
        if isinstance(ocr_switch, dict):  # 如果是字典，那么就是自定义OCR参数
            ocr_switch = copy.deepcopy(llm_kwargs)
            ocr_switch.update(ocr_switch)
        vision_submission = reader_fns.submit_threads_img_handle(img_mapping, save_path, cor_cache, ocr_switch)
        filed_sum = 0
        for t in vision_submission:
            base_name = os.path.basename(t)
            try:
                img_content, img_path, status = vision_submission[t].result()

                vision_loading_statsu.update({base_name: img_content})
                chatbot[-1][1] = gpt_old_say + vision_format(f'检测到识图开关为`{ocr_switch}`，正在识别图片中的文字...',
                                                             vision_loading_statsu)
                yield from update_ui(chatbot=chatbot, history=history)
                if not status or status == '本次识别结果读取数据库缓存':  # 出现异常，不替换文本
                    content = content.replace(img_mapping[t],
                                              f'[{img_mapping[t]}]\n{base_name}图片识别结果：\n{img_content}')
                else:
                    filed_sum += 1
                    logger.warning(f'{img_mapping[t]} 识别失败，跳过，error: {status}')
            except Exception as e:
                filed_sum += 1
                status = f'`{t}` `{trimmed_format_exc()}` 识别失败，过滤这个图片\n\n'
                vision_loading_statsu.update({base_name: status})  # 错误展示完整路径
                chatbot[-1][1] = gpt_old_say + vision_format(f'啊哦，有文件失败了哦', vision_loading_statsu)
                yield from update_ui(chatbot=chatbot, history=history)

        chatbot[-1][1] = gpt_old_say + vision_format(
            f'图片识别完成, 共{len(vision_submission)}张图片，识别失败`{filed_sum}`', vision_loading_statsu, 'Done')
        yield from update_ui(chatbot=chatbot, history=history, msg='Done')
    return content.replace(init_path.base_path, 'file=.')  # 增加保障，防止路径泄露
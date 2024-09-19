# encoding: utf-8
# @Time   : 2024/7/19
# @Author : Spike
# @Descr   :
import copy
import os
from common.func_box import valid_img_extensions
from common import gr_converter_html
from common.func_box import extract_link_pf, long_name_processing
from common.toolbox import update_ui,  trimmed_format_exc
from common.path_handler import init_path
from crazy_functions import reader_fns
from common.logger_handler import logger
from crazy_functions.submit_fns import json_args_return


def write_markdown(data, hosts, file_name):
    """
    Args: 将data写入md文件
        data: 数据
        hosts: 用户标识
        file_name: 另取文件名
    Returns: 写入的文件地址
    """
    user_path = os.path.join(init_path.private_files_path, hosts, 'markdown')
    os.makedirs(user_path, exist_ok=True)
    md_file = os.path.join(user_path, f"{file_name}.md")
    with open(file=md_file, mode='w', encoding='utf-8') as f:
        f.write(data)
    return md_file


def file_classification_to_dict(gpt_response_collection):
    """
    接收gpt多线程的返回数据，将输入相同的作为key, gpt返回以列表形式添加到对应的key中，主要是为了区分不用文件的输入
    Args:
        gpt_response_collection: 多线程GPT的返回耶
    Returns: {'文件': [结果1， 结果2...], '文件2': [结果1， 结果2...]}
    """
    file_classification = {}
    for inputs, you_say in zip(gpt_response_collection[1::2], gpt_response_collection[0::2]):
        file_classification[you_say] = []
    for inputs, you_say in zip(gpt_response_collection[1::2], gpt_response_collection[0::2]):
        file_classification[you_say].append(inputs)
    return file_classification


def batch_recognition_images_to_md(img_list, ipaddr):
    """
    Args: 将图片批量识别然后写入md文件
        img_list: 图片地址list
        ipaddr: 用户所属标识
    Returns: [文件list]
    """
    temp_list = []
    save_path = os.path.join(init_path.private_files_path, ipaddr, 'ocr_to_md')
    for img in img_list:
        if os.path.exists(img):
            img_content, img_result, _ = reader_fns.ImgHandler(img, save_path).get_paddle_ocr()
            temp_file = os.path.join(save_path,
                                     img_content.splitlines()[0][:20] + '.md')
            with open(temp_file, mode='w', encoding='utf-8') as f:
                f.write(f"{gr_converter_html.html_view_blank(temp_file)}\n\n" + img_content)
            temp_list.append(temp_list)
        else:
            print(img, '文件路径不存在')
    return temp_list


def name_de_add_sort(response, index=0):
    if type(index) is not int:
        return response  # 如果不是数字下标，那么不排序
    try:
        unique_tuples = set(tuple(lst) for lst in response)
        de_result = [list(tpl) for tpl in unique_tuples]
        d = {}
        for i, v in enumerate(de_result):
            if len(v) >= index:
                if v[index] not in d:
                    d[v[index]] = i
            else:
                d[v[len(v)]] = i
        de_result.sort(key=lambda x: d[x[index]])
        return de_result
    except:
        from common.toolbox import trimmed_format_exc
        tb_str = '```\n' + trimmed_format_exc() + '```'
        print(tb_str)
        return response


def parsing_json_in_text(txt_data: list, old_case, filter_list: list = 'None----', tags='插件补充的用例', sort_index=0):
    response = []
    desc = '\n\n---\n\n'.join(txt_data)
    for index in range(len(old_case)):
        # 获取所有Json
        supplementary_data = reader_fns.MdProcessor(txt_data[index]).json_to_list()
        # 兼容一下哈
        if len(txt_data) != len(old_case): index = -1
        # 过滤掉产出带的表头数据
        filtered_supplementary_data = [data for data in supplementary_data
                                       if filter_list[:5] != data[:5] or filter_list[-5:] != data[-5:]]
        # 检查 filtered_supplementary_data 是否为空
        if not filtered_supplementary_data:
            max_length = 0  # 或其他合适的默认值
        else:
            max_length = max(len(lst) for lst in filtered_supplementary_data)
        supplement_temp_data = [lst + [''] * (max_length - len(lst)) for lst in filtered_supplementary_data]
        for new_case in supplement_temp_data:
            if new_case not in old_case[index] and new_case + [tags] not in old_case[index]:
                old_case[index].append(new_case + [tags])
        response.extend(old_case[index])
    # 按照名称排列重组
    response = name_de_add_sort(response, sort_index)
    return response, desc


def result_extract_to_test_cases(gpt_response_collection, llm_kwargs, plugin_kwargs, chatbot, history):
    """
    Args:
        gpt_response_collection: [输入文件标题， 输出]
        llm_kwargs: 调优参数
        plugin_kwargs: 插件调优参数
        chatbot: 对话组件
        history: 对话历史
        file_key: 存入历史文件
    Returns: None
    """
    template_file, sheet, sort_index = json_args_return(plugin_kwargs,
                                                        ['写入指定模版', '写入指定Sheet', '用例下标排序'])
    file_classification = file_classification_to_dict(gpt_response_collection)
    chat_file_list = ''
    you_say = '准备将测试用例写入Excel中...'
    chatbot.append([you_say, chat_file_list])
    yield from update_ui(chatbot, history)
    files_limit = {}
    for file_name in file_classification:
        # 处理md数据
        test_case = reader_fns.MdProcessor(file_classification[file_name]).tabs_to_list()
        sort_test_case = name_de_add_sort(test_case, sort_index)
        # 正式准备写入文件
        save_path = os.path.join(init_path.private_files_path, llm_kwargs['ipaddr'], 'test_case')
        xlsx_handler = reader_fns.XlsxHandler(template_file, output_dir=save_path, sheet=sheet)
        xlsx_handler.split_merged_cells()  # 先把合并的单元格拆分，避免写入失败
        file_path = xlsx_handler.list_write_to_excel(sort_test_case, save_as_name=long_name_processing(file_name))
        chat_file_list += f'生成结果如下:\t {gr_converter_html.html_view_blank(__href=file_path, to_tabs=True)}\n\n'
        chatbot[-1] = [you_say, chat_file_list]
        yield from update_ui(chatbot, history)
        files_limit.update({file_path: file_name})
    return files_limit


def result_supplementary_to_test_case(gpt_response_collection, llm_kwargs, plugin_kwargs, chatbot, history):
    template_file, sheet, sort_index = json_args_return(plugin_kwargs,
                                                        ['写入指定模版', '写入指定Sheet', '用例下标排序'])
    if not sheet:
        sheet, = json_args_return(plugin_kwargs, ['读取指定Sheet'])
    file_classification = file_classification_to_dict(gpt_response_collection)
    chat_file_list = ''
    you_say = '准备将测试用例写入Excel中...'
    chatbot.append([you_say, chat_file_list])
    yield from update_ui(chatbot, history)
    files_limit = {}
    for file_name in file_classification:
        old_file = plugin_kwargs['上阶段文件']
        old_case = plugin_kwargs[old_file].get('原测试用例数据', [])
        header = plugin_kwargs[old_file].get('原测试用例表头', [])
        test_case, desc = parsing_json_in_text(file_classification[file_name], old_case, filter_list=header,
                                               sort_index=sort_index)
        save_path = os.path.join(init_path.private_files_path, llm_kwargs['ipaddr'], 'test_case')
        # 写入excel
        xlsx_handler = reader_fns.XlsxHandler(template_file, output_dir=save_path, sheet=sheet)
        show_file_name = long_name_processing(file_name)
        file_path = xlsx_handler.list_write_to_excel(test_case, save_as_name=show_file_name)
        # 写入 markdown
        md_path = os.path.join(save_path, f"{show_file_name}.md")
        reader_fns.MdHandler(md_path).save_markdown(desc)
        chat_file_list += f'{show_file_name}生成结果如下:\t {gr_converter_html.html_view_blank(__href=file_path, to_tabs=True)}\n\n' \
                          f'{show_file_name}补充思路如下：\t{gr_converter_html.html_view_blank(__href=md_path, to_tabs=True)}\n\n---\n\n'
        chatbot[-1] = [you_say, chat_file_list]
        yield from update_ui(chatbot, history)
        files_limit.update({file_path: file_name})
    return files_limit


def result_converter_to_flow_chart(gpt_response_collection, llm_kwargs, plugin_kwargs, chatbot, history):
    """
    Args: 将输出结果写入md，并转换为流程图
        gpt_response_collection: [输入、输出]
        llm_kwargs: 调优参数
        chatbot: 对话组件
        history: 历史对话
    Returns:
        None
    """
    file_classification = file_classification_to_dict(gpt_response_collection)
    file_limit = {}
    chat_file_list = ''
    you_say = '请将Markdown结果转换为流程图~'
    chatbot.append([you_say, chat_file_list])
    for file_name in file_classification:
        inputs_count = ''
        for value in file_classification[file_name]:
            inputs_count += str(value).replace('```', '')  # 去除头部和尾部的代码块, 避免流程图堆在一块
        save_path = os.path.join(init_path.private_files_path, llm_kwargs['ipaddr'])
        md_file = os.path.join(save_path, f"{long_name_processing(file_name)}.md")
        html_file = reader_fns.MdHandler(md_path=md_file, output_dir=save_path).save_mark_map()
        chat_file_list += "View: " + gr_converter_html.html_view_blank(md_file, to_tabs=True) + \
                          '\n\n--- \n\n View: ' + gr_converter_html.html_view_blank(html_file)
        chatbot.append([you_say, chat_file_list])
        yield from update_ui(chatbot=chatbot, history=history, msg='成功写入文件！')
        file_limit.update({md_file: file_name})
    # f'tips: 双击空白处可以放大～\n\n' f'{html_iframe_code(html_file=html)}'  无用，不允许内嵌网页了
    return file_limit


def result_written_to_markdown(gpt_response_collection, llm_kwargs, plugin_kwargs, chatbot, history, stage=''):
    """
    Args: 将输出结果写入md
        gpt_response_collection: [输入、输出]
        llm_kwargs: 调优参数
        chatbot: 对话组件
        history: 历史对话
    Returns:
        None
    """
    file_classification = file_classification_to_dict(gpt_response_collection)
    file_limit = []
    chat_file_list = ''
    you_say = '请将Markdown结果写入文件中...'
    chatbot.append([you_say, chat_file_list])
    for file_name in file_classification:
        inputs_all = ''
        for value in file_classification[file_name]:
            inputs_all += value
        md = write_markdown(data=inputs_all, hosts=llm_kwargs['ipaddr'],
                            file_name=long_name_processing(file_name) + stage)
        chat_file_list = f'markdown已写入文件，下次使用插件可以直接提交markdown文件啦 {gr_converter_html.html_view_blank(md, to_tabs=True)}'
        chatbot[-1] = [you_say, chat_file_list]
        yield from update_ui(chatbot=chatbot, history=history, msg='成功写入文件！')
        file_limit.append(md)
    return file_limit



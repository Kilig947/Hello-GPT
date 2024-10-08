import importlib
import time
import inspect
import re
import importlib
import inspect
from common.history_handler import thread_write_chat_json, get_user_basedata
import base64
import gradio as gr
import math
from latex2mathml.converter import convert as tex2mathml
from functools import wraps, lru_cache
import shutil
import os
import time
import glob
import sys
import threading

from common.func_box import (num_tokens_from_string, user_client_mark,
                             created_atime, encryption_str, extract_link_pf, valid_img_extensions)
from common.path_handler import init_path

############################### 插件输入输出接驳区 #######################################
import gradio
import shutil
import glob
import logging
import uuid
from functools import wraps
from shared_utils.config_loader import get_conf
from shared_utils.config_loader import set_conf
from shared_utils.config_loader import set_multi_conf
from shared_utils.config_loader import read_single_conf_with_lru_cache
from shared_utils.advanced_markdown_format import format_io
from shared_utils.advanced_markdown_format import markdown_convertion
from shared_utils.key_pattern_manager import select_api_key
from shared_utils.key_pattern_manager import is_any_api_key
from shared_utils.key_pattern_manager import what_keys
from shared_utils.connect_void_terminal import get_chat_handle
from shared_utils.connect_void_terminal import get_plugin_handle
from shared_utils.connect_void_terminal import get_plugin_default_kwargs
from shared_utils.connect_void_terminal import get_chat_default_kwargs
from shared_utils.text_mask import apply_gpt_academic_string_mask
from shared_utils.text_mask import build_gpt_academic_masked_string
from shared_utils.text_mask import apply_gpt_academic_string_mask_langbased
from shared_utils.text_mask import build_gpt_academic_masked_string_langbased
from shared_utils.map_names import map_friendly_names_to_model
from shared_utils.map_names import map_model_to_friendly_names
from shared_utils.map_names import read_one_api_model_name
from shared_utils.handle_upload import extract_archive
from typing import List

pj = os.path.join
default_user_name = "default_user"

"""
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
第一部分
函数插件输入输出接驳区
    - ChatBotWithCookies:   带Cookies的Chatbot类，为实现更多强大的功能做基础
    - ArgsGeneralWrapper:   装饰器函数，用于重组输入参数，改变输入参数的顺序与结构
    - update_ui:            刷新界面用 yield from update_ui(chatbot, history)
    - CatchException:       将插件中出的所有问题显示在界面上
    - HotReload:            实现插件的热更新
    - trimmed_format_exc:   打印traceback，为了安全而隐藏绝对地址
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
"""


class ChatBotWithCookies(list):

    def __init__(self, cookie):
        self._cookies = cookie

    def write_list(self, list):
        for t in list:
            self.append(t)

    def get_list(self):
        return [t for t in self]

    def get_cookies(self):
        return self._cookies

    def get_user(self):
        return self._cookies.get("user_name", default_user_name)


def end_predict(chatbot, history, llm_kwargs):
    count_time = round(time.time() - llm_kwargs['start_time'], 3)
    count_tokens = num_tokens_from_string(listing=history)
    status = f"本次对话耗时: `{count_time}s` \t 本次对话使用tokens: `{count_tokens}`"
    yield from update_ui(chatbot=chatbot, history=history, msg=status, end_code=1)  # 刷新界面


def ArgsGeneralWrapper(func):
    """
    装饰器函数ArgsGeneralWrapper，用于重组输入参数，改变输入参数的顺序与结构。
    该装饰器是大多数功能调用的入口。
    函数示意图：https://mermaid.live/edit#pako:eNqNVFtPGkEY_StkntoEDQtLoTw0sWqapjQxVWPabmOm7AiEZZcsQ9QiiW012qixqdeqqIn10geBh6ZR8PJnmAWe-hc6l3VhrWnLEzNzzvnO953ZyYOYoSIQAWOaMR5LQBN7hvoU3UN_g5iu7imAXEyT4wUF3Pd0dT3y9KGYYUJsmK8V0GPGs0-QjkyojZgwk0Fm82C2dVghX08U8EaoOHjOfoEMU0XmADRhOksVWnNLjdpM82qFzB6S5Q_WWsUhuqCc3JtAsVR_OoMnhyZwXgHWwbS1d4gnsLVZJp-P6mfVxveqAgqC70Jz_pQCOGDKM5xFdNNPDdilF6uSU_hOYqu4a3MHYDZLDzq5fodrC3PWcEaFGPUaRiqJWK_W9g9rvRITa4dhy_0nw67SiePMp3oSR6PPn41DGgllkvkizYwsrmtaejTFd8V4yekGmT1zqrt4XGlAy8WTuiPULF01LksZvukSajfQQRAxmYi5S0D81sDcyzapVdn6sYFHkjhhGyel3frVQnvsnbR23lEjlhIlaOJiFPWzU5G4tfNJo8ejwp47-TbvJkKKZvmxA6SKo16oaazJysfG6klr9T0pbTW2ZqzlL_XaT8fYbQLXe4mSmvoCZXMaa7FePW6s7jVqK9bujvse3WFjY5_Z4KfsA4oiPY4T7Drvn1tLJTbG1to1qR79ulgk89-oJbvZzbIwJty6u20LOReWa9BvwserUd9s9MIKc3x5TUWEoAhUyJK5y85w_yG-dFu_R9waoU7K581y8W_qLle35-rG9Nxcrz8QHRsc0K-r9NViYRT36KsFvCCNzDRMqvSVyzOKAnACpZECIvSvCs2UAhS9QHEwh43BST0GItjMIS_I8e-sLwnj9A262cxA_ZVh0OUY1LJiDSJ5MAEiUijYLUtBORR6KElyQPaCSRDpksNSd8AfluSgHPaFC17wjrOlbgbzyyFf4IFPDvoD_sJvnkdK-g
    """

    def decorated(cookies, max_length, llm_model, txt,  # 调优参数
                  top_p, temperature, n_choices, stop_sequence,
                  max_context, max_generation, presence_penalty,
                  frequency_penalty, logit_bias, user_identifier, response_format,
                  chatbot, history, system_prompt, plugin_advanced_arg,
                  # 输入栏模式
                  single_mode, agent_mode,
                  # 知识库
                  kb_selects, vector_score, vector_top_k,
                  # 高级设置
                  models, history_num, worker_num, ocr_trust,
                  # 个人信息配置
                  openai_key, wps_cookies, qq_cookies, feishu_header,
                  project_user_key, project_header,
                  ipaddr: gr.Request, *args):  # 参数来源__main__.py self.input_combo
        """"""
        start_time = time.time()
        real_llm = {
            'llm_model': llm_model,
            'top_p': top_p, 'temperature': temperature, 'n_choices': n_choices, 'stop': stop_sequence,
            'max_context': max_context, 'max_generation': max_generation, 'presence_penalty': presence_penalty,
            'frequency_penalty': frequency_penalty, 'logit_bias': logit_bias, 'user_identifier': user_identifier,
            'response_format': response_format, 'input_models': models,
            'system_prompt': system_prompt, 'ipaddr': user_client_mark(ipaddr),
            'kb_config': {"names": kb_selects, 'score': vector_score, 'top-k': vector_top_k},
        }
        llm_kwargs = {  # 这些不会写入对话记录哦
            **real_llm, 'api_key': cookies.get('api_key') + f",{openai_key}",
            'worker_num': worker_num, 'start_time': start_time, 'ocr': ocr_trust,
            'max_length': max_length,
            'wps_cookies': wps_cookies, 'qq_cookies': qq_cookies, 'feishu_header': feishu_header,
            'project_config': {
                'project_user_key': project_user_key, 'project_header': project_header
            }
        }
        # 历史对话轮次
        history = history[0:history_num * 2]  # 为了保证历史记录永远是偶数
        # 对话参数
        if not cookies.get('first_chat') and args:
            cookies['first_chat'] = args[0] + "_" + created_atime()
        plugin_kwargs = {
            "advanced_arg": plugin_advanced_arg,
            "parameters_def": ''
        }
        cookies.update({**real_llm})
        # 这里的cookie是引用，所以后面赋值会同步到chatbot中，所以后续传cookie还是chatbot.get_cookies()都是一样的
        chatbot_with_cookie = ChatBotWithCookies(cookies)
        # 引入一个有cookie的chatbot
        chatbot_with_cookie.write_list(chatbot)
        # 根据args判断需要对提交和历史对话做什么处理
        txt_proc, history, func_redirect = yield from plugins_selection(txt, history, plugin_kwargs,
                                                                        args, cookies, chatbot_with_cookie, llm_kwargs,
                                                                        func)
        # 根据提交处理器判断需要对提交做什么处理
        txt_proc, func_redirect = yield from model_selection(txt_proc, models, llm_kwargs, plugin_kwargs, cookies,
                                                             chatbot_with_cookie, history, args, func_redirect)
        # 根据cookie 或 对话配置决定到底走哪一步
        yield from func_decision_tree(func_redirect, cookies, single_mode, agent_mode,
                                      txt_proc, llm_kwargs, plugin_kwargs, chatbot_with_cookie,
                                      history, system_prompt, args)
        # 将对话记录写入文件
        yield from end_predict(chatbot_with_cookie, history, llm_kwargs)
        threading.Thread(target=thread_write_chat_json,
                         args=(chatbot_with_cookie, user_client_mark(ipaddr))).start()

    return decorated


def model_selection(txt, models, llm_kwargs, plugin_kwargs, cookies, chatbot_with_cookie, history, args, func):
    txt_proc = txt
    # 开关调整
    if 'OCR缓存' in models: llm_kwargs.update({'ocr_cache': True})
    if '关联缺陷' in models: llm_kwargs['project_config'].update({'关联缺陷': True})
    if '关联用例' in models: llm_kwargs['project_config'].update({'关联用例': True})
    if '关联任务' in models: llm_kwargs['project_config'].update({'关联任务': True})
    if 'Vision-Img' in models:
        if isinstance(plugin_kwargs.get('advanced_arg'), dict):
            if not plugin_kwargs['advanced_arg'].get('开启OCR'):
                plugin_kwargs['advanced_arg']['开启OCR'] = _vision_select_model(llm_kwargs, models)
        elif isinstance(plugin_kwargs.get('parameters_def'), dict):
            if not plugin_kwargs['parameters_def'].get('开启OCR'):
                plugin_kwargs['parameters_def']['开启OCR'] = _vision_select_model(llm_kwargs, models)
        else:
            plugin_kwargs.update({'开启OCR': _vision_select_model(llm_kwargs, models)})

    # 实实在在会改变输入的
    if 'input加密' in models: txt_proc = encryption_str(txt_proc)
    if len(args) == 0 or 'RetryChat' in args and not cookies.get('is_plugin'):
        if '网络链接RAG' in models and 'moonshot' not in llm_kwargs.get('llm_model', ''):
            from crazy_functions.submit_fns import user_input_embedding_content, check_url_domain_cloud, \
                submit_no_use_ui_task
            valid_types = ['pdf', 'md', 'xlsx', 'docx']
            wps_links, qq_link, feishu_link, project_link = check_url_domain_cloud(txt_proc)
            local_file = extract_link_pf(txt_proc, valid_types)
            if wps_links or qq_link or feishu_link or local_file or project_link:  # 提前检测，有文件才进入下一步文件处理
                yield from update_ui(chatbot_with_cookie, history, msg='检测到提交存在文档链接，正在跳转Reader...')
                input_embedding_content = yield from user_input_embedding_content(txt_proc, chatbot_with_cookie,
                                                                                  history, llm_kwargs, plugin_kwargs,
                                                                                  valid_types)
                txt_proc_embedding = "\n\n---\n\n".join(
                    [v for i, v in enumerate(input_embedding_content) if i % 2 == 1])
                if txt_proc_embedding:
                    txt_proc = txt_proc_embedding
                    func = submit_no_use_ui_task
            else:
                yield from update_ui(chatbot_with_cookie, history, msg='Switching to normal dialog...')
        img_info = extract_link_pf(txt, valid_img_extensions)
        if 'vision' not in llm_kwargs['llm_model'] and img_info:
            vision_llm = _vision_select_model(llm_kwargs, models)
            if vision_llm:
                llm_kwargs.update(vision_llm)
            yield from update_ui(chatbot_with_cookie, history,
                                 msg=f'Switching to `{llm_kwargs["llm_model"]}` dialog...')
    return txt_proc, func


def _vision_select_model(llm_kwargs, models):
    vision_llm = {}
    if llm_kwargs['llm_model'].startswith('gpt') and "4o" not in llm_kwargs['llm_model']:
        if "gpt4-v自动识图" in models:
            vision_llm['llm_model'] = 'gpt-4-vision-preview'
    elif llm_kwargs['llm_model'].startswith('gemini'):
        if "gemini-v自动识图" in models:
            vision_llm['llm_model'] = 'gemini-pro-vision'
    elif llm_kwargs['llm_model'].startswith('glm'):
        if "glm-v自动识图" in models:
            vision_llm['llm_model'] = 'glm-4v'
    else:
        return False
    return vision_llm


def plugins_selection(txt_proc, history, plugin_kwargs, args, cookies, chatbot_with_cookie, llm_kwargs, func):
    # 插件会传多参数，如果是插件，那么更新知识库 和 默认高级参数
    if len(args) > 1:
        plugin_kwargs['advanced_arg'] = ''
        plugin_kwargs.update({'parameters_def': args[1]})
        cookies['is_plugin'] = {'func_name': args[0], 'input': txt_proc, 'kwargs': plugin_kwargs}
    elif len(args) == 1 and 'RetryChat' not in args:
        history = history[:-2]  # 不采取重试的对话历史
        cookies['is_plugin'] = {'func_name': args[0], 'input': txt_proc, 'kwargs': plugin_kwargs}
    elif len(args) == 0 or 'RetryChat' in args and not cookies.get('is_plugin'):
        cookies['is_plugin'] = False
        plugin_kwargs['advanced_arg'] = ''
        from common.knowledge_base.kb_func import vector_recall_by_input
        if llm_kwargs['kb_config']['names']:
            from crazy_functions.submit_fns import submit_no_use_ui_task
            unpacking_input = yield from vector_recall_by_input(txt_proc, chatbot_with_cookie, history,
                                                                llm_kwargs, '知识库提示词',
                                                                '引用知识库回答')
            txt_proc, history = unpacking_input
            func = submit_no_use_ui_task
    return txt_proc, history, func


def func_decision_tree(func, cookies, single_mode, agent_mode,
                       txt_proc, llm_kwargs, plugin_kwargs, chatbot_with_cookie,
                       history, system_prompt, args):
    if single_mode:
        deci_history = []
    else:
        deci_history = history
    if cookies.get('lock_plugin', None) is None:
        is_try = args[0] if 'RetryChat' in args else None
        if is_try:
            user_data = get_user_basedata(chatbot_with_cookie, llm_kwargs['ipaddr'])
            plugin = user_data['chat'][-1].get('plugin')
            if plugin:
                txt_proc = plugin['input']
                from common.crazy_functional import crazy_fns
                func_name = plugin['func_name']
                plugin_kwargs.update(plugin['kwargs'])
                cookies['is_plugin'] = {'func_name': func_name, 'input': txt_proc, 'kwargs': plugin_kwargs}
                try_f = crazy_fns.get(func_name, False)
                if try_f: try_f = try_f['Function']
            else:
                txt_proc = cookies.get('last_chat', '')
                from crazy_functions.submit_fns import submit_no_use_ui_task
                try_f = submit_no_use_ui_task
                args = ()
            yield from try_f(txt_proc, llm_kwargs, plugin_kwargs, chatbot_with_cookie,
                             deci_history, system_prompt, *args)
        else:
            if agent_mode:
                from common.crazy_functional import crazy_fns
                plugin_agent = crazy_fns['插件代理助手']['Function']
                func = plugin_agent
            yield from func(txt_proc, llm_kwargs, plugin_kwargs, chatbot_with_cookie,
                            deci_history, system_prompt, *args)
        cookies.update({'last_chat': txt_proc})
    else:
        # 处理少数情况下的特殊插件的锁定状态
        module, fn_name = cookies['lock_plugin'].split('->')
        f_hot_reload = getattr(importlib.import_module(module, fn_name), fn_name)
        yield from f_hot_reload(txt_proc, llm_kwargs, plugin_kwargs, chatbot_with_cookie, history, system_prompt,
                                *args)
        # 判断一下用户是否错误地通过对话通道进入，如果是，则进行提醒
        final_cookies = chatbot_with_cookie.get_cookies()
        # len(args) != 0 代表“提交”键对话通道，或者基础功能通道
        if len(args) != 0 and 'files_to_promote' in final_cookies and len(final_cookies['files_to_promote']) > 0:
            chatbot_with_cookie.append(
                ["检测到**滞留的缓存文档**，请及时处理。", "请及时点击“**保存当前对话**”获取所有滞留文档。"])
            yield from update_ui(chatbot_with_cookie, final_cookies['history'], msg="检测到被滞留的缓存文档")


def update_ui(chatbot, history, msg=None, end_code=0, *args):  # 刷新界面
    """
    刷新用户界面
    """
    assert isinstance(chatbot, ChatBotWithCookies), "在传递chatbot的过程中不要将其丢弃。必要时，可用clear将其清空，然后用for+append循环重新赋值。"
    cookies = chatbot.get_cookies()
    # 备份一份History作为记录
    cookies.update({"history": history})
    # 解决插件锁定时的界面显示问题
    if cookies.get("lock_plugin", None):
        label = (
                cookies.get("llm_model", "")
                + " | "
                + "正在锁定插件"
                + cookies.get("lock_plugin", None)
        )
        chatbot_gr = gradio.update(value=chatbot, label=label)
        if cookies.get("label", "") != label:
            cookies["label"] = label  # 记住当前的label
    elif cookies.get("label", None):
        chatbot_gr = gradio.update(value=chatbot, label=cookies.get("llm_model", ""))
        cookies["label"] = None  # 清空label
    else:
        chatbot_gr = chatbot
    if not msg:
        msg = gr.update()
    event = [cookies, chatbot_gr, history, msg]
    if end_code:
        yield event + [gr.update(visible=False), gr.update(visible=True)]
    else:
        yield event + [gr.update(visible=True), gr.update(visible=False)]


def update_ui_lastest_msg(lastmsg: str, chatbot: ChatBotWithCookies, history: list, delay=1):  # 刷新界面
    """
    刷新用户界面
    """
    if len(chatbot) == 0:
        chatbot.append(["update_ui_last_msg", lastmsg])
    chatbot[-1] = list(chatbot[-1])
    chatbot[-1][-1] = lastmsg
    yield from update_ui(chatbot=chatbot, history=history)
    time.sleep(delay)


def trimmed_format_exc():
    import os, traceback

    str = traceback.format_exc()
    current_path = os.getcwd()
    replace_path = "../.."
    return str.replace(current_path, replace_path)


def CatchException(f):
    """
    装饰器函数，捕捉函数f中的异常并封装到一个生成器中返回，并显示到聊天当中。
    """

    @wraps(f)
    def decorated(main_input: str, llm_kwargs: dict, plugin_kwargs: dict,
                  chatbot_with_cookie: ChatBotWithCookies, history: list, *args, **kwargs):
        try:
            yield from f(main_input, llm_kwargs, plugin_kwargs, chatbot_with_cookie, history, args, kwargs)
        except Exception as e:
            tb_str = '```\n' + trimmed_format_exc() + '```'
            if len(chatbot_with_cookie) == 0:
                chatbot_with_cookie.clear()
                chatbot_with_cookie.append(["插件调度异常", "异常原因"])
            chatbot_with_cookie[-1][1] += f"\n\n[Local Message] 插件调用出错: \n\n{tb_str} \n"
            yield from update_ui(chatbot=chatbot_with_cookie, history=history, msg=f'异常 {e}')  # 刷新界面

    return decorated


def HotReload(f):
    """
    HotReload的装饰器函数，用于实现Python函数插件的热更新。
    函数热更新是指在不停止程序运行的情况下，更新函数代码，从而达到实时更新功能。
    在装饰器内部，使用wraps(f)来保留函数的元信息，并定义了一个名为decorated的内部函数。
    内部函数通过使用importlib模块的reload函数和inspect模块的getmodule函数来重新加载并获取函数模块，
    然后通过getattr函数获取函数名，并在新模块中重新加载函数。
    最后，使用yield from语句返回重新加载过的函数，并在被装饰的函数上执行。
    最终，装饰器函数返回内部函数。这个内部函数可以将函数的原始定义更新为最新版本，并执行函数的新版本。
    """
    if get_conf("PLUGIN_HOT_RELOAD"):

        @wraps(f)
        def decorated(*args, **kwargs):
            fn_name = f.__name__
            f_hot_reload = getattr(importlib.reload(inspect.getmodule(f)), fn_name)
            yield from f_hot_reload(*args, **kwargs)

        return decorated
    else:
        return f


####################################### 其他小工具 #####################################

"""
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
第二部分
其他小工具:
    - write_history_to_file:    将结果写入markdown文件中
    - regular_txt_to_markdown:  将普通文本转换为Markdown格式的文本。
    - report_exception:         向chatbot中添加简单的意外错误信息
    - text_divide_paragraph:    将文本按照段落分隔符分割开，生成带有段落标签的HTML代码。
    - markdown_convertion:      用多种方式组合，将markdown转化为好看的html
    - format_io:                接管gradio默认的markdown处理方式
    - on_file_uploaded:         处理文件的上传（自动解压）
    - on_report_generated:      将生成的报告自动投射到文件上传区
    - clip_history:             当历史上下文过长时，自动截断
    - get_conf:                 获取设置
    - select_api_key:           根据当前的模型类别，抽取可用的api-key
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
"""


def get_reduce_token_percent(text: str):
    """
    * 此函数未来将被弃用
    """
    try:
        # text = "maximum context length is 4097 tokens. However, your messages resulted in 4870 tokens"
        pattern = r"(\d+)\s+tokens\b"
        match = re.findall(pattern, text)
        EXCEED_ALLO = 500  # 稍微留一点余地，否则在回复时会因余量太少出问题
        max_limit = float(match[0]) - EXCEED_ALLO
        current_tokens = float(match[1])
        ratio = max_limit / current_tokens
        assert ratio > 0 and ratio < 1
        return ratio, str(int(current_tokens - max_limit))
    except:
        return 0.5, "不详"


def write_history_to_file(
        history: list, file_basename: str = None, file_fullname: str = None, auto_caption: bool = True
):
    """
    将对话记录history以Markdown格式写入文件中。如果没有指定文件名，则使用当前时间生成文件名。
    """
    import os
    if file_fullname is None:
        if file_basename is not None:
            file_fullname = pj(get_log_folder(), file_basename)
        else:
            file_fullname = pj(get_log_folder(), f"GPT-Academic-{gen_time_str()}.md")
    os.makedirs(os.path.dirname(file_fullname), exist_ok=True)
    with open(file_fullname, "w", encoding="utf8") as f:
        f.write("# GPT-Academic Report\n")
        for i, content in enumerate(history):
            try:
                if type(content) != str:
                    content = str(content)
            except:
                continue
            if i % 2 == 0 and auto_caption:
                f.write("## ")
            try:
                f.write(content)
            except:
                # remove everything that cannot be handled by utf8
                f.write(content.encode("utf-8", "ignore").decode())
            f.write("\n\n")
    res = os.path.abspath(file_fullname)
    return res


def regular_txt_to_markdown(text: str):
    """
    将普通文本转换为Markdown格式的文本。
    """
    text = text.replace("\n", "\n\n")
    text = text.replace("\n\n\n", "\n\n")
    text = text.replace("\n\n\n", "\n\n")
    return text


def report_exception(chatbot: ChatBotWithCookies, history: list, a: str, b: str):
    """
    向chatbot中添加错误信息
    """
    chatbot.append([1])
    history.extend([a, b])


def find_free_port() -> int:
    """
    返回当前系统中可用的未使用端口。
    """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def find_recent_files(directory: str) -> List[str]:
    """
    Find files that is created with in one minutes under a directory with python, write a function
    """
    import os
    import time

    current_time = time.time()
    one_minute_ago = current_time - 60
    recent_files = []
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    for filename in os.listdir(directory):
        file_path = pj(directory, filename)
        if file_path.endswith(".log"):
            continue
        created_time = os.path.getmtime(file_path)
        if created_time >= one_minute_ago:
            if os.path.isdir(file_path):
                continue
            recent_files.append(file_path)

    return recent_files


def file_already_in_downloadzone(file: str, user_path: str):
    try:
        parent_path = os.path.abspath(user_path)
        child_path = os.path.abspath(file)
        if os.path.samefile(os.path.commonpath([parent_path, child_path]), parent_path):
            return True
        else:
            return False
    except:
        return False


def promote_file_to_downloadzone(file: str, rename_file: str = None, chatbot: ChatBotWithCookies = None):
    # 将文件复制一份到下载区
    import shutil

    if chatbot is not None:
        user_name = get_user(chatbot)
    else:
        user_name = default_user_name
    if not os.path.exists(file):
        raise FileNotFoundError(f"文件{file}不存在")
    user_path = get_log_folder(user_name, plugin_name=None)
    if file_already_in_downloadzone(file, user_path):
        new_path = file
    else:
        user_path = get_log_folder(user_name, plugin_name="downloadzone")
        if rename_file is None:
            rename_file = f"{gen_time_str()}-{os.path.basename(file)}"
        new_path = pj(user_path, rename_file)
        # 如果已经存在，先删除
        if os.path.exists(new_path) and not os.path.samefile(new_path, file):
            os.remove(new_path)
        # 把文件复制过去
        if not os.path.exists(new_path):
            shutil.copyfile(file, new_path)
    # 将文件添加到chatbot cookie中
    if chatbot is not None:
        if "files_to_promote" in chatbot._cookies:
            current = chatbot._cookies["files_to_promote"]
        else:
            current = []
        if new_path not in current:  # 避免把同一个文件添加多次
            chatbot._cookies.update({"files_to_promote": [new_path] + current})
    return new_path


def disable_auto_promotion(chatbot: ChatBotWithCookies):
    chatbot._cookies.update({"files_to_promote": []})
    return


def del_outdated_uploads(outdate_time_seconds: float, target_path_base: str = None):
    if target_path_base is None:
        user_upload_dir = get_conf("PATH_PRIVATE_UPLOAD")
    else:
        user_upload_dir = target_path_base
    current_time = time.time()
    one_hour_ago = current_time - outdate_time_seconds
    # Get a list of all subdirectories in the user_upload_dir folder
    # Remove subdirectories that are older than one hour
    for subdirectory in glob.glob(f"{user_upload_dir}/*"):
        subdirectory_time = os.path.getmtime(subdirectory)
        if subdirectory_time < one_hour_ago:
            try:
                shutil.rmtree(subdirectory)
            except:
                pass
    return


def on_file_uploaded(files, chatbot, txt, cookies, ipaddr: gr.Request):
    from crazy_functions.reader_fns.local_markdown import to_markdown_tabs
    from common.gr_converter_html import file_manifest_filter_type
    private_upload = init_path.private_files_path.replace(init_path.base_path, '.')
    #     shutil.rmtree('./private_upload/')  不需要删除文件

    if type(ipaddr) is str:
        ipaddr = ipaddr
    else:
        ipaddr = user_client_mark(ipaddr)
    time_tag = created_atime()
    time_tag_path = os.path.join(private_upload, ipaddr, 'temp', time_tag)
    os.makedirs(f'{time_tag_path}', exist_ok=True)
    err_msg = ''
    for file in files:
        file_origin_name = os.path.basename(file.orig_name)
        new_file = os.path.join(time_tag_path, file_origin_name)
        shutil.copy(file.name, new_file)
        err_msg += extract_archive(f'{time_tag_path}/{file_origin_name}',
                                   dest_dir=f'{time_tag_path}/{file_origin_name}.extract')
    moved_files = [fp for fp in glob.glob(f'{time_tag_path}/**/*', recursive=True)]
    moved_view = [[i] for i in moved_files]
    moved_files_str = to_markdown_tabs(head=['Preview' for i in moved_files], tabs=moved_view)
    if type(chatbot) is str:
        if not txt:
            txt = {'file_path': '', 'know_name': '', 'know_obj': {}, 'file_list': []}
        txt.update({'file_path': time_tag_path})
    else:
        txt += "\n".join(file_manifest_filter_type(moved_files, md_type=True))
        cookies.update({
            'most_recent_uploaded': {
                'path': f'{time_tag_path}',
                'time': time.time(),
                'time_str': time_tag
            }})
    return chatbot, txt


def on_report_generated2(request: gradio.Request, files: List[str], chatbot: ChatBotWithCookies,
                         txt: str, txt2: str, checkboxes: List[str], cookies: dict):
    from crazy_functions.reader_fns.local_markdown import to_markdown_tabs
    if 'file_to_promote' in cookies:
        report_files = cookies['file_to_promote']
        cookies.pop('file_to_promote')

    # 移除过时的旧文件从而节省空间&保护隐私
    outdate_time_seconds = 60
    del_outdated_uploads(outdate_time_seconds)
    user = user_client_mark(request)
    # 创建工作路径
    user_name = "default" if not user else 'default'
    time_tag = gen_time_str()
    target_path_base = get_upload_folder(user_name, tag=time_tag)
    os.makedirs(target_path_base, exist_ok=True)

    # 移除过时的旧文件从而节省空间&保护隐私
    outdate_time_seconds = 3600  # 一小时
    del_outdated_uploads(outdate_time_seconds, get_upload_folder(user_name))

    # 逐个文件转移到目标路径
    upload_msg = ""
    for file in files:
        file_origin_name = os.path.basename(file.orig_name)
        this_file_path = pj(target_path_base, file_origin_name)
        shutil.move(file.name, this_file_path)
        upload_msg += extract_archive(
            file_path=this_file_path, dest_dir=this_file_path + ".extract"
        )

    # 整理文件集合 输出消息
    files = glob.glob(f"{target_path_base}/**/*", recursive=True)
    moved_files = [fp for fp in files]
    max_file_to_show = 10
    if len(moved_files) > max_file_to_show:
        moved_files = moved_files[:max_file_to_show // 2] + [
            f'... ( 📌省略{len(moved_files) - max_file_to_show}个文件的显示 ) ...'] + \
                      moved_files[-max_file_to_show // 2:]
    moved_files_str = to_markdown_tabs(head=["文件"], tabs=[moved_files], omit_path=target_path_base)
    chatbot.append(
        [
            "我上传了文件，请查收",
            f"[Local Message] 收到以下文件 （上传到路径：{target_path_base}）: " +
            f"\n\n{moved_files_str}" +
            f"\n\n调用路径参数已自动修正到: \n\n{txt}" +
            f"\n\n现在您点击任意函数插件时，以上文件将被作为输入参数" +
            upload_msg,
        ]
    )

    txt, txt2 = target_path_base, ""
    if "浮动输入区" in checkboxes:
        txt, txt2 = txt2, txt

    # 记录近期文件
    cookies.update(
        {
            "most_recent_uploaded": {
                "path": target_path_base,
                "time": time.time(),
                "time_str": time_tag,
            }
        }
    )
    return chatbot, txt, txt2, cookies


def on_report_generated(cookies: dict, files: List[str], chatbot: ChatBotWithCookies):
    if "files_to_promote" in cookies:
        report_files = cookies["files_to_promote"]
        cookies.pop("files_to_promote")
    else:
        report_files = []
    if len(report_files) == 0:
        return cookies, None, chatbot
    file_links = ""
    for f in report_files:
        file_links += (
            f'<br/><a href="file={os.path.abspath(f)}" target="_blank">{f}</a>'
        )
    chatbot.append(["报告如何远程获取？", f"报告已经添加到右侧“文件上传区”（可能处于折叠状态），请查收。{file_links}"])
    return cookies, report_files, chatbot


def load_chat_cookies():
    API_KEY, LLM_MODEL, AZURE_API_KEY = get_conf(
        "API_KEY", "LLM_MODEL", "AZURE_API_KEY"
    )
    AZURE_CFG_ARRAY, NUM_CUSTOM_BASIC_BTN = get_conf(
        "AZURE_CFG_ARRAY", "NUM_CUSTOM_BASIC_BTN"
    )

    # deal with azure openai key
    if is_any_api_key(AZURE_API_KEY):
        if is_any_api_key(API_KEY):
            API_KEY = API_KEY + "," + AZURE_API_KEY
        else:
            API_KEY = AZURE_API_KEY
    if len(AZURE_CFG_ARRAY) > 0:
        for azure_model_name, azure_cfg_dict in AZURE_CFG_ARRAY.items():
            if not azure_model_name.startswith("azure"):
                raise ValueError("AZURE_CFG_ARRAY中配置的模型必须以azure开头")
            AZURE_API_KEY_ = azure_cfg_dict["AZURE_API_KEY"]
            if is_any_api_key(AZURE_API_KEY_):
                if is_any_api_key(API_KEY):
                    API_KEY = API_KEY + "," + AZURE_API_KEY_
                else:
                    API_KEY = AZURE_API_KEY_

    customize_fn_overwrite_ = {}
    for k in range(NUM_CUSTOM_BASIC_BTN):
        customize_fn_overwrite_.update(
            {
                "自定义按钮"
                + str(k + 1): {
                    "Title": r"",
                    "Prefix": r"请在自定义菜单中定义提示词前缀.",
                    "Suffix": r"请在自定义菜单中定义提示词后缀",
                }
            }
        )
    return {
        "api_key": API_KEY,
        "llm_model": LLM_MODEL,
        "customize_fn_overwrite": customize_fn_overwrite_,
    }


def clear_line_break(txt):
    txt = txt.replace("\n", " ")
    txt = txt.replace("  ", " ")
    txt = txt.replace("  ", " ")
    return txt


class DummyWith:
    """
    这段代码定义了一个名为DummyWith的空上下文管理器，
    它的作用是……额……就是不起作用，即在代码结构不变得情况下取代其他的上下文管理器。
    上下文管理器是一种Python对象，用于与with语句一起使用，
    以确保一些资源在代码块执行期间得到正确的初始化和清理。
    上下文管理器必须实现两个方法，分别为 __enter__()和 __exit__()。
    在上下文执行开始的情况下，__enter__()方法会在代码块被执行前被调用，
    而在上下文执行结束时，__exit__()方法则会被调用。
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return


def run_gradio_in_subpath(demo, auth, port, custom_path):
    """
    把gradio的运行地址更改到指定的二次路径上
    """

    def is_path_legal(path: str) -> bool:
        """
        check path for sub url
        path: path to check
        return value: do sub url wrap
        """
        if path == "/":
            return True
        if len(path) == 0:
            print(
                "ilegal custom path: {}\npath must not be empty\ndeploy on root url".format(
                    path
                )
            )
            return False
        if path[0] == "/":
            if path[1] != "/":
                print("deploy on sub-path {}".format(path))
                return True
            return False
        print(
            "ilegal custom path: {}\npath should begin with '/'\ndeploy on root url".format(
                path
            )
        )
        return False

    if not is_path_legal(custom_path):
        raise RuntimeError("Ilegal custom path")
    import uvicorn
    import gradio as gr
    from fastapi import FastAPI

    app = FastAPI()
    if custom_path != "/":
        @app.get("/")
        def read_main():
            return {"message": f"Gradio is running at: {custom_path}"}

    app = gr.mount_gradio_app(app, demo, path=custom_path)
    uvicorn.run(app, host="0.0.0.0", port=port)  # , auth=auth


def clip_history(inputs, history, tokenizer, max_token_limit):
    """
    reduce the length of history by clipping.
    this function search for the longest entries to clip, little by little,
    until the number of token of history is reduced under threshold.
    通过裁剪来缩短历史记录的长度。
    此函数逐渐地搜索最长的条目进行剪辑，
    直到历史记录的标记数量降低到阈值以下。
    """
    import numpy as np
    from request_llms.bridge_all import model_info

    def get_token_num(txt):
        return len(tokenizer.encode(txt, disallowed_special=()))

    input_token_num = get_token_num(inputs)

    if max_token_limit < 5000:
        output_token_expect = 256  # 4k & 2k models
    elif max_token_limit < 9000:
        output_token_expect = 512  # 8k models
    else:
        output_token_expect = 1024  # 16k & 32k models

    if input_token_num < max_token_limit * 3 / 4:
        # 当输入部分的token占比小于限制的3/4时，裁剪时
        # 1. 把input的余量留出来
        max_token_limit = max_token_limit - input_token_num
        # 2. 把输出用的余量留出来
        max_token_limit = max_token_limit - output_token_expect
        # 3. 如果余量太小了，直接清除历史
        if max_token_limit < output_token_expect:
            history = []
            return history
    else:
        # 当输入部分的token占比 > 限制的3/4时，直接清除历史
        history = []
        return history

    everything = [""]
    everything.extend(history)
    n_token = get_token_num("\n".join(everything))
    everything_token = [get_token_num(e) for e in everything]

    # 截断时的颗粒度
    delta = max(everything_token) // 16

    while n_token > max_token_limit:
        where = np.argmax(everything_token)
        encoded = tokenizer.encode(everything[where], disallowed_special=())

    clipped_encoded = encoded[: len(encoded) - delta]
    everything[where] = tokenizer.decode(clipped_encoded)[
                        :-1
                        ]  # -1 to remove the may-be illegal char
    everything_token[where] = get_token_num(everything[where])
    n_token = get_token_num("\n".join(everything))

    history = everything[1:]
    return history


"""
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
第三部分
其他小工具:
    - zip_folder:    把某个路径下所有文件压缩，然后转移到指定的另一个路径中（gpt写的）
    - gen_time_str:  生成时间戳
    - ProxyNetworkActivate: 临时地启动代理网络（如果有）
    - objdump/objload: 快捷的调试函数
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
"""


def zip_folder(source_folder, dest_folder, zip_name):
    import zipfile
    import os

    # Make sure the source folder exists
    if not os.path.exists(source_folder):
        print(f"{source_folder} does not exist")
        return

    # Make sure the destination folder exists
    if not os.path.exists(dest_folder):
        print(f"{dest_folder} does not exist")
        return

    # Create the name for the zip file
    zip_file = pj(dest_folder, zip_name)

    # Create a ZipFile object
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the source folder and add files to the zip file
        for foldername, subfolders, filenames in os.walk(source_folder):
            for filename in filenames:
                filepath = pj(foldername, filename)
                zipf.write(filepath, arcname=os.path.relpath(filepath, source_folder))

    # Move the zip file to the destination folder (if it wasn't already there)
    if os.path.dirname(zip_file) != dest_folder:
        os.rename(zip_file, pj(dest_folder, os.path.basename(zip_file)))
        zip_file = pj(dest_folder, os.path.basename(zip_file))

    print(f"Zip file created at {zip_file}")


def zip_result(folder):
    t = gen_time_str()
    zip_folder(folder, get_log_folder(), f"{t}-result.zip")
    return pj(get_log_folder(), f"{t}-result.zip")


def gen_time_str():
    import time

    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def get_log_folder(user=default_user_name, plugin_name="shared"):
    if user is None:
        user = default_user_name
    PATH_LOGGING = get_conf("PATH_LOGGING")
    if plugin_name is None:
        _dir = pj(PATH_LOGGING, user)
    else:
        _dir = pj(PATH_LOGGING, user, plugin_name)
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return _dir


def get_upload_folder(user=default_user_name, tag=None):
    PATH_PRIVATE_UPLOAD = get_conf("PATH_PRIVATE_UPLOAD")
    if user is None:
        user = default_user_name
    if tag is None or len(tag) == 0:
        target_path_base = pj(PATH_PRIVATE_UPLOAD, user)
    else:
        target_path_base = pj(PATH_PRIVATE_UPLOAD, user, tag)
    return target_path_base


def is_the_upload_folder(string):
    PATH_PRIVATE_UPLOAD = get_conf("PATH_PRIVATE_UPLOAD")
    pattern = r"^PATH_PRIVATE_UPLOAD[\\/][A-Za-z0-9_-]+[\\/]\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$"
    pattern = pattern.replace("PATH_PRIVATE_UPLOAD", PATH_PRIVATE_UPLOAD)
    if re.match(pattern, string):
        return True
    else:
        return False


def get_user(chatbotwithcookies: ChatBotWithCookies):
    return chatbotwithcookies._cookies.get("user_name", default_user_name)


class ProxyNetworkActivate:
    """
    这段代码定义了一个名为ProxyNetworkActivate的空上下文管理器, 用于给一小段代码上代理
    """

    def __init__(self, task=None) -> None:
        self.task = task
        if not task:
            # 不给定 task，那么我们默认代理生效
            self.valid = True
        else:

            WHEN_TO_USE_PROXY = get_conf("WHEN_TO_USE_PROXY")
            self.valid = task in WHEN_TO_USE_PROXY

    def __enter__(self):
        if not self.valid:
            return self

        proxies = get_conf("proxies")
        if "no_proxy" in os.environ:
            os.environ.pop("no_proxy")
        if proxies is not None:
            if "http" in proxies:
                os.environ["HTTP_PROXY"] = proxies["http"]
            if "https" in proxies:
                os.environ["HTTPS_PROXY"] = proxies["https"]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        os.environ["no_proxy"] = "*"
        if "HTTP_PROXY" in os.environ:
            os.environ.pop("HTTP_PROXY")
        if "HTTPS_PROXY" in os.environ:
            os.environ.pop("HTTPS_PROXY")
        return


def objdump(obj, file="objdump.tmp"):
    import pickle

    with open(file, "wb+") as f:
        pickle.dump(obj, f)
    return


def objload(file="objdump.tmp"):
    import pickle, os

    if not os.path.exists(file):
        return
    with open(file, "rb") as f:
        return pickle.load(f)


def Singleton(cls):
    """
    一个单实例装饰器
    """
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton


def get_pictures_list(path):
    file_manifest = [f for f in glob.glob(f"{path}/**/*.jpg", recursive=True)]
    file_manifest += [f for f in glob.glob(f"{path}/**/*.jpeg", recursive=True)]
    file_manifest += [f for f in glob.glob(f"{path}/**/*.png", recursive=True)]
    return file_manifest


def have_any_recent_upload_image_files(chatbot: ChatBotWithCookies):
    _5min = 5 * 60
    if chatbot is None:
        return False, None  # chatbot is None
    most_recent_uploaded = chatbot._cookies.get("most_recent_uploaded", None)
    if not most_recent_uploaded:
        return False, None  # most_recent_uploaded is None
    if time.time() - most_recent_uploaded["time"] < _5min:
        most_recent_uploaded = chatbot._cookies.get("most_recent_uploaded", None)
        path = most_recent_uploaded["path"]
        file_manifest = get_pictures_list(path)
        if len(file_manifest) == 0:
            return False, None
        return True, file_manifest  # most_recent_uploaded is new
    else:
        return False, None  # most_recent_uploaded is too old


# Claude3 model supports graphic context dialogue, reads all images
def every_image_file_in_path(chatbot: ChatBotWithCookies):
    if chatbot is None:
        return False, []  # chatbot is None
    most_recent_uploaded = chatbot._cookies.get("most_recent_uploaded", None)
    if not most_recent_uploaded:
        return False, []  # most_recent_uploaded is None
    path = most_recent_uploaded["path"]
    file_manifest = get_pictures_list(path)
    if len(file_manifest) == 0:
        return False, []
    return True, file_manifest


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_max_token(llm_kwargs):
    from request_llms.bridge_all import model_info

    return model_info[llm_kwargs["llm_model"]]["max_token"]


def check_packages(packages=[]):
    import importlib.util

    for p in packages:
        spam_spec = importlib.util.find_spec(p)
        if spam_spec is None:
            raise ModuleNotFoundError


def map_file_to_sha256(file_path):
    import hashlib

    with open(file_path, 'rb') as file:
        content = file.read()

    # Calculate the SHA-256 hash of the file contents
    sha_hash = hashlib.sha256(content).hexdigest()

    return sha_hash


def check_repeat_upload(new_pdf_path, pdf_hash):
    '''
    检查历史上传的文件是否与新上传的文件相同，如果相同则返回(True, 重复文件路径)，否则返回(False，None)
    '''
    import PyPDF2

    user_upload_dir = os.path.dirname(os.path.dirname(new_pdf_path))
    file_name = os.path.basename(new_pdf_path)

    file_manifest = [f for f in glob.glob(f'{user_upload_dir}/**/{file_name}', recursive=True)]

    for saved_file in file_manifest:
        with open(new_pdf_path, 'rb') as file1, open(saved_file, 'rb') as file2:
            reader1 = PyPDF2.PdfFileReader(file1)
            reader2 = PyPDF2.PdfFileReader(file2)

            # 比较页数是否相同
            if reader1.getNumPages() != reader2.getNumPages():
                continue

            # 比较每一页的内容是否相同
            for page_num in range(reader1.getNumPages()):
                page1 = reader1.getPage(page_num).extractText()
                page2 = reader2.getPage(page_num).extractText()
                if page1 != page2:
                    continue

        maybe_project_dir = glob.glob('{}/**/{}'.format(get_log_folder(), pdf_hash + ".tag"), recursive=True)

        if len(maybe_project_dir) > 0:
            return True, os.path.dirname(maybe_project_dir[0])

    # 如果所有页的内容都相同，返回 True
    return False, None


def log_chat(llm_model: str, input_str: str, output_str: str):
    try:
        if output_str and input_str and llm_model:
            uid = str(uuid.uuid4().hex)
            logging.info(f"[Model({uid})] {llm_model}")
            input_str = input_str.rstrip('\n')
            logging.info(f"[Query({uid})]\n{input_str}")
            output_str = output_str.rstrip('\n')
            logging.info(f"[Response({uid})]\n{output_str}\n\n")
    except:
        print(trimmed_format_exc())

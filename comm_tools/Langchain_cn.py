import os.path
from comm_tools import toolbox
from crazy_functions import crazy_utils
import gradio as gr
from comm_tools import func_box
from crazy_functions.crazy_utils import knowledge_archive_interface
from crazy_functions import crazy_box

def knowledge_base_writing(links: str, select, name, kai_handle, ipaddr: gr.Request):
    # < --------------------读取参数--------------- >
    vector_path = os.path.join(func_box.knowledge_path, ipaddr.client.host)
    if name and select != '新建知识库':
        os.rename(os.path.join(vector_path, select), os.path.join(vector_path, name))
        kai_id = name
    elif name and select == '新建知识库': kai_id = name
    elif select and select != '新建知识库': kai_id = select
    else: kai_id = func_box.created_atime()
    yield '开始咯开始咯～', '', gr.Dropdown.update(), kai_handle
    files = kai_handle['file_path']
    # < --------------------读取文件--------------- >
    file_manifest = []
    spl,  = toolbox.get_conf('spl')
    # 本地文件
    error = ''
    for sp in spl:
        _, file_manifest_tmp, _ = crazy_utils.get_files_from_everything(files, type=f'.{sp}')
        file_manifest += file_manifest_tmp
        try:
            _, kdocs_manifest_tmp, _ = crazy_box.get_kdocs_from_everything(links, type=sp)
        except:
            import traceback
            error_str = traceback.format_exc()
            error += f'提取出错文件错误啦\n\n```\n{error_str}\n```'
            yield (f"", error, gr.Dropdown.update(), kai_handle)
            kdocs_manifest_tmp = []
        file_manifest += kdocs_manifest_tmp
    if len(file_manifest) == 0:
        types = "\t".join(f"`{s}`" for s in spl)
        yield (f'没有找到任何可读取文件， 当前支持解析的文件格式包括: \n\n{types}', error,
               gr.Dropdown.update(),  kai_handle)
        return
    # < -------------------预热文本向量化模组--------------- >
    yield ('正在加载向量化模型...', '', gr.Dropdown.update(), kai_handle)
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    with toolbox.ProxyNetworkActivate():    # 临时地激活代理网络
        HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")
    # < -------------------构建知识库--------------- >
    preprocessing_files = func_box.to_markdown_tabs(head=['文件'], tabs=[file_manifest])
    yield (f'正在准备将以下文件向量化，生成知识库文件：\n\n{preprocessing_files}', error,
           gr.Dropdown.update(), kai_handle)
    with toolbox.ProxyNetworkActivate():    # 临时地激活代理网络
        kai = knowledge_archive_interface(vs_path=vector_path)
        qa_handle, vs_path = kai.construct_vector_store(vs_id=kai_id, files=file_manifest)
    kai_files = kai.get_loaded_file()
    kai_handle['file_list'] = [os.path.basename(file) for file in kai_files]
    kai_files = func_box.to_markdown_tabs(head=['文件'], tabs=[kai_files])
    kai_handle['know_obj'].update({kai_id: qa_handle})
    kai_handle['know_name'] = kai_id
    yield (f'构建完成, 当前知识库内有效的文件如下, 已自动帮您选中知识库，现在你可以畅快的开始提问啦～\n\n{kai_files}', error,
           gr.Dropdown.update(value='新建知识库', choices=obtain_a_list_of_knowledge_bases(ipaddr)),  kai_handle)


def knowledge_base_query(txt, kai_id, chatbot, history, llm_kwargs, args, ipaddr: gr.Request):
    # < -------------------为空时，不去查询向量数据库--------------- >
    if not txt: return txt
    # < -------------------检索Prompt--------------- >
    new_txt = f'{txt}'
    for id in kai_id:   #
        if llm_kwargs['know_dict']['know_obj'].get(id, False):
            kai = llm_kwargs['know_dict']['know_obj'][id]
        else:
            kai = knowledge_archive_interface(vs_path=os.path.join(func_box.knowledge_path, ipaddr.client.host))
        # < -------------------查询向量数据库--------------- >
        chatbot.append([txt, f'正在将问题向量化，然后对{func_box.html_tag_color(id)}知识库进行匹配'])
        yield from toolbox.update_ui(chatbot=chatbot, history=history)  # 刷新界面
        resp, prompt, _ok = kai.answer_with_archive_by_id(txt, id)
        if resp:
            referenced_documents = "\n".join([f"{k}: " + doc.page_content for k, doc in enumerate(resp['source_documents'])])
            new_txt += f'\n以下三个引号内的是知识库提供的参考文档：\n"""\n{referenced_documents}\n"""'
    return new_txt


def obtain_a_list_of_knowledge_bases(ipaddr):
    def get_directory_list(folder_path):
        directory_list = []
        for root, dirs, files in os.walk(folder_path):
            for dir_name in dirs:
                directory_list.append(dir_name)
        return directory_list
    user_path = os.path.join(func_box.knowledge_path, ipaddr.client.host)
    return get_directory_list(user_path) + get_directory_list(func_box.knowledge_path_sys_path)


def obtaining_knowledge_base_files(vs_id, chatbot, kai_handle, ipaddr: gr.Request):
    from crazy_functions.crazy_utils import knowledge_archive_interface
    if vs_id:
        if isinstance(chatbot, toolbox.ChatBotWithCookies):
            pass
        else:
            chatbot = toolbox.ChatBotWithCookies(chatbot)
            chatbot.write_list(chatbot)
        chatbot.append([None, f'正在检查知识库内文件{"  ".join([func_box.html_tag_color(i)for i in vs_id])}'])
        yield chatbot, gr.Column.update(visible=False), '🏃🏻‍ 正在努力轮询中....请稍等， tips：知识库可以多选，但不要贪杯哦～️', kai_handle
        kai_files = {}
        for id in vs_id:
            kai = knowledge_archive_interface(vs_path=os.path.join(func_box.knowledge_path, ipaddr.client.host))
            qa_handle, _dict = kai.get_init_file(vs_id=id)
            kai_files.update(_dict)
            kai_handle['know_obj'].update({id: qa_handle})
        tabs = [[_id, func_box.html_view_blank(file), kai_files[file][_id]] for file in kai_files for _id in kai_files[file]]
        kai_handle['file_list'] = [os.path.basename(file) for file in kai_files]
        chatbot.append([None, f'检查完成，当前选择的知识库内可用文件如下：'
                              f'\n\n {func_box.to_markdown_tabs(head=["所属知识库", "文件", "文件类型"], tabs=tabs)}\n\n'
                              f'🤩 快来向我提问吧～'])

        yield chatbot, gr.Column.update(visible=False), '✅ 检查完成', kai_handle
    else:
        yield chatbot, gr.update(), 'Done', kai_handle


def knowledge_base_assisted_questioning(txt, kai_id, chatbot, history, llm_kwargs, ipaddr: gr.Request):
    pass

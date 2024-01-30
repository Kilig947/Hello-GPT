# encoding: utf-8
# @Time   : 2023/9/16
# @Author : Spike
# @Descr   :
import gradio as gr
from common import func_box, toolbox
from webui_elem import webui_local

i18n = webui_local.I18nAuto()
get_html = func_box.get_html


def popup_title(txt):
    with gr.Row():
        gr.Markdown(txt)
        gr.HTML(get_html("close_btn.html").format(
            obj="box"), elem_classes="close-btn")


class Settings:

    def __init__(self):
        pass

    def _draw_setting_senior(self):
        with gr.Tab(label=i18n("高级")):
            self.models_box = gr.CheckboxGroup(choices=['input加密', '预加载知识库', 'OCR缓存'], value=['input加密'],
                                               label="提交开关").style(container=False)
            worker_num = toolbox.get_conf('DEFAULT_WORKER_NUM')
            self.default_worker_num = gr.Slider(minimum=1, maximum=30, value=worker_num, step=1,
                                                show_label=True, interactive=True, label="插件多线程最大并行",
                                                ).style(container=False)
            self.pro_tf_slider = gr.Slider(minimum=1, maximum=200, value=15, step=1, interactive=True,
                                           label="搜索展示详细字符", show_label=True).style(container=False)
            self.ocr_identifying_trust = gr.Slider(minimum=0.01, maximum=1.0, value=0.60, step=0.01,
                                                   interactive=True, show_label=True,
                                                   label="Paddleocr OCR 识别信任指数").style(container=False)
            self.secret_css, self.secret_font = gr.Textbox(visible=False), gr.Textbox(visible=False)
            AVAIL_THEMES, latex_option = toolbox.get_conf('AVAIL_THEMES', 'latex_option')
            self.theme_dropdown = gr.Dropdown(AVAIL_THEMES, value=AVAIL_THEMES[0], label=i18n("更换UI主题"),
                                              interactive=True, allow_custom_value=True,
                                              info='更多主题, 请查阅Gradio主题商店: '
                                                   'https://huggingface.co/spaces/gradio/theme-gallery',
                                              ).style(container=False)
            self.latex_option = gr.Dropdown(latex_option, value=latex_option[0], label=i18n("更换Latex输出格式"),
                                            interactive=True, ).style(container=False)
            gr.HTML(get_html("appearance_switcher.html").format(
                label=i18n("切换亮暗色主题")), elem_classes="insert-block", visible=False)
            self.single_turn_checkbox = gr.Checkbox(label=i18n(
                "无记忆对话"), value=False, elem_classes="switch-checkbox",
                elem_id="gr-single-session-cb", visible=False)

    def _darw_private_operation(self):
        with gr.TabItem('个人中心', id='private'):
            with gr.Row(elem_id='about-tab'):
                gr.Markdown('#### 粉身碎骨浑不怕 要留清白在人间\n\n'
                            '`这里的东西只有你自己能看，不要告诉别人哦`\n\n' \
                            + func_box.html_tag_color('我们不会保存你的个人信息，页面刷新后这里的信息就会被丢弃',
                                                      color='rgb(227 179 51)'))
            self.usageTxt = gr.Markdown(i18n(
                "**发送消息** 或 **提交key** 以显示额度"), elem_id="usage-display",
                elem_classes="insert-block", visible=False)
            self.openai_keys = gr.Textbox(
                show_label=True, placeholder=f"Your OpenAi-API-key...",
                # value=hide_middle_chars(user_api_key.value),
                type="password",  # visible=not HIDE_MY_KEY,
                label="API-Key").style(container=False)
            self.wps_cookie = gr.Textbox(lines=3, label='WPS Cookies', type='password',
                                         placeholder=f"Your WPS cookies json...", ).style(container=False)
            self.qq_cookie = gr.Textbox(lines=3, label='QQ Cookies', type='password',
                                        placeholder=f"Your QQ cookies json...", ).style(container=False)
            self.feishu_cookie = gr.Textbox(lines=3, label='Feishu Header', type='password',
                                            placeholder=f"Your Feishu header json...", ).style(container=False)
            with gr.Row():
                self.info_perish_btn = gr.Button('清除我来过的痕迹', variant='stop',
                                                 full_width=True, elem_classes='danger_btn')
                self.exit_login_btn = gr.LogoutButton(icon='', link='/logout')

    def _draw_setting_info(self):
        APPNAME = toolbox.get_conf('APPNAME')
        with gr.Tab(label=i18n("关于"), elem_id="about-tab"):
            gr.Markdown("# " + i18n(APPNAME))
            gr.HTML(get_html("footer.html").format(versions=''), elem_id="footer")
            gr.Markdown('', elem_id="description")

    def draw_popup_settings(self):
        with gr.Box(elem_id="chuanhu-setting"):
            popup_title("## " + i18n("设置"))
            with gr.Tabs(elem_id="chuanhu-setting-tabs"):
                self._draw_setting_senior()
                self._darw_private_operation()
                self._draw_setting_info()


class AdvancedSearch:

    def __init__(self):
        pass

    def draw_popup_search(self):
        with gr.Box(elem_id="spike-search"):
            popup_title("## " + i18n("高级搜索"))
            with gr.Box():
                with gr.Row():
                    self.history_search_txt = gr.Textbox(show_label=False, elem_classes='search_txt',
                                                         placeholder="输入你想要搜索的对话记录", container=False)
                with gr.Row(elem_classes='search-example'):
                    self.pro_history_state = gr.State({'samples': None})
                    self.pro_history_list = gr.Dataset(components=[gr.HTML(visible=False)], samples_per_page=10,
                                                       visible=False, label='搜索结果',
                                                       samples=[[". . ."] for i in range(20)], type='index')


class Config:

    def __init__(self):
        pass

    def draw_popup_config(self):
        with gr.Box(elem_id="web-config", visible=False):
            gr.HTML(get_html('web_config.html').format(
                enableCheckUpdate_config='',
                hideHistoryWhenNotLoggedIn_config='',
                forView_i18n=i18n("仅供查看"),
                deleteConfirm_i18n_pref=i18n("你真的要"),
                deleteConfirm_i18n_suff=i18n(" 吗？"),
                usingLatest_i18n=i18n("您使用的就是最新版！"),
                updatingMsg_i18n=i18n("正在尝试更新..."),
                updateSuccess_i18n=i18n("更新成功，请重启本程序"),
                updateFailure_i18n=i18n(
                    '更新失败，请尝试<a href="https://github.com/GaiZhenbiao/ChuanhuChatGPT/wiki/使用教程#手动更新" target="_blank">手动更新</a>'),
                regenerate_i18n=i18n("重新生成"),
                deleteRound_i18n=i18n("删除这轮问答"),
                renameChat_i18n=i18n("重命名该对话"),
                validFileName_i18n=i18n("请输入有效的文件名，不要包含以下特殊字符："),
            ))


class Prompt:

    def __init__(self):
        pass

    def _draw_tabs_prompt(self):
        self.devs_document = toolbox.get_conf('devs_document')
        with gr.TabItem('提示词', id='prompt'):
            Tips = "用 BORF 分析法设计GPT 提示词:\n" \
                   "1、阐述背景 B(Background): 说明背景，为chatGPT提供充足的信息\n" \
                   "2、定义目标 O(Objectives):“我们希望实现什么”\n" \
                   "3、定义关键结果 R(key Result):“我要什么具体效果”\n" \
                   "4、试验并调整，改进 E(Evolve):三种改进方法自由组合\n" \
                   "\t 改进输入：从答案的不足之处着手改进背景B,目标O与关键结果R\n" \
                   "\t 改进答案：在后续对话中指正chatGPT答案缺点\n" \
                   "\t 重新生成：尝试在`提示词`不变的情况下多次生成结果，优中选优\n" \
                   "\t 熟练使用占位符{{{v}}}:  当`提示词`存在占位符，则优先将{{{v}}}替换为预期文本"
            self.pro_edit_txt = gr.Textbox(show_label=False, lines=12,
                                           elem_classes='no_padding_input',
                                           placeholder=Tips)
            with gr.Row():
                with gr.Column(elem_classes='column_left'):
                    with gr.Accordion('Prompt Upload', open=False):
                        self.pro_upload_btn = gr.File(file_count='single', file_types=['.yaml', '.json'],
                                                      label=f'上传你的提示词文件, 编写格式请遵循上述开发者文档', )
                    self.prompt_status = gr.Markdown(value='')
                with gr.Column(elem_classes='column_right'):
                    with gr.Row():
                        self.prompt_cls_select = gr.Dropdown(choices=[], value='',
                                                             label='提示词分类', elem_classes='normal_select',
                                                             allow_custom_value=True, interactive=True, container=False
                                                             )
                        self.pro_name_txt = gr.Textbox(show_label=False, placeholder='提示词名称', container=False)
                    with gr.Row():
                        self.pro_del_btn = gr.Button("删除提示词", size='sm', full_width=True)
                        self.pro_new_btn = gr.Button("保存提示词", variant="primary", size='sm', full_width=True)

    def _draw_tabs_masks(self):
        with gr.TabItem('Prompt Masks 🎭', id='masks'):
            def_sys = i18n('你是一个xxx角色，你会xxx技能，你将按照xxx要求，回答我的问题')
            self.masks_dataset = gr.Dataframe(value=[['system', def_sys]], datatype='str',
                                              headers=['role', 'content'], col_count=(2, 'fixed'),
                                              interactive=True, show_label=False, row_count=(1, "dynamic"),
                                              wrap=True, type='array', elem_id='mask_tabs')
            self.masks_delete_btn = gr.Button('Del New row', size='sm', elem_id='mk_del')
            self.masks_clear_btn = gr.Button(value='Clear All', size='sm', elem_id='mk_clear')
            with gr.Row():
                with gr.Column(elem_classes='column_left'):
                    with gr.Accordion('Chatbot Preview', open=False):
                        self.mask_preview_chat = gr.Chatbot(label='', show_label=False)
                    self.mask_status = gr.Markdown(value='')
                with gr.Column(elem_classes='column_right'):
                    with gr.Row():
                        self.mask_cls_select = gr.Dropdown(choices=[], value='',
                                                           label='Masks分类', elem_classes='normal_select',
                                                           allow_custom_value=True, interactive=True, container=False
                                                           )
                        self.masks_name_txt = gr.Textbox(show_label=False, placeholder='Mask名称', container=False)
                    with gr.Row():
                        self.masks_del_btn = gr.Button("删除Mask", size='sm', full_width=True)
                        self.masks_new_btn = gr.Button("保存Mask", variant="primary", size='sm', full_width=True)

    def _draw_langchain_base(self):
        spl = toolbox.get_conf('spl')
        with gr.TabItem('知识库构建', id='langchain_tab', elem_id='langchain_tab'):
            with gr.Row():
                with gr.Column(elem_classes='column_left'):
                    self.langchain_upload = gr.Files(label="支持解析多类型文档，多文件建议使用zip上传",
                                                     file_count="multiple", file_types=spl)
                    self.langchain_links = gr.Textbox(show_label=False, placeholder='网络文件,多个链接使用换行间隔',
                                                      elem_classes='no_padding_input')
                    self.langchain_know_kwargs = gr.State(
                        {'file_path': '', 'know_name': '', 'know_obj': {}, 'file_list': []})
                    #  file_path 是上传文件存储的地址，know_name，know_obj是ql向量化后的对象
                with gr.Column(elem_classes='column_right'):
                    with gr.Row():
                        self.langchain_classifi = gr.Dropdown(choices=[], value="知识库", interactive=True,
                                                              label="选择知识库分类", allow_custom_value=True,
                                                              elem_classes='normal_select', container=False)
                        self.langchain_cls_name = gr.Textbox(show_label=False, placeholder='已有知识库重命名',
                                                             container=False,
                                                             visible=False)
                    with gr.Row():
                        self.langchain_select = gr.Dropdown(choices=[], value=r"", allow_custom_value=True,
                                                            interactive=True, label="新建or增量重构",
                                                            elem_classes='normal_select', container=False)
                        self.langchain_name = gr.Textbox(show_label=False, placeholder='已有知识库重命名',
                                                         container=False,
                                                         visible=False)
                    with gr.Row():
                        self.langchain_submit = gr.Button(value='构建/更新知识库', variant='primary', size='sm')
                        self.langchain_stop = gr.Button(value='停止构建', size='sm')
            func_box.md_division_line()
            self.langchain_status = gr.Markdown(value='')
            self.langchain_error = gr.Markdown(value='')

    def _draw_popup_training(self):
        with gr.TabItem('OpenAi' + i18n('预训练'), id='training_tab', elem_id='training_tab'):
            self.openai_train_status = gr.Markdown(label=i18n("训练状态"), value=i18n(
                "查看[使用介绍](https://github.com/GaiZhenbiao/ChuanhuChatGPT/wiki/使用教程#微调-gpt-35)"))
            with gr.Row():
                with gr.Column(elem_classes='column_left'):
                    self.dataset_selection = gr.Files(label=i18n("选择数据集"), file_types=[
                        ".xlsx", ".jsonl"], file_count="single")
                    self.dataset_preview_json = gr.JSON(
                        label=i18n("数据集预览"), readonly=True)
                    self.upload_to_openai_btn = gr.Button(
                        i18n("上传到OpenAI"), variant="primary", interactive=False)
                with gr.Column(elem_classes='column_right'):
                    self.openai_ft_file_id = gr.Textbox(label=i18n(
                        "文件ID"), value="", lines=1, placeholder=i18n("上传到 OpenAI 后自动填充"))
                    self.openai_ft_suffix = gr.Textbox(label=i18n(
                        "模型名称后缀"), value="", lines=1, placeholder=i18n("可选，用于区分不同的模型"))
                    self.openai_train_epoch_slider = gr.Slider(label=i18n(
                        "训练轮数（Epochs）"), minimum=1, maximum=100, value=3, step=1, interactive=True)
                    self.openai_start_train_btn = gr.Button(
                        i18n("开始训练"), variant="primary", interactive=False)
                    self.openai_status_refresh_btn = gr.Button(i18n("刷新状态"))
                    self.openai_cancel_all_jobs_btn = gr.Button(
                        i18n("取消所有任务"))
                    self.add_to_models_btn = gr.Button(
                        i18n("添加训练好的模型到模型列表"), interactive=False)

    def draw_popup_prompt(self):
        with gr.Box(elem_id="spike-prompt"):
            popup_title("### " + i18n(f"百宝袋"))
            with gr.Tabs(elem_id="treasure-bag") as self.treasure_bag_tab:
                self._draw_tabs_prompt()
                self._draw_tabs_masks()
                self._draw_langchain_base()
                self._draw_popup_training()



class FakeComponents:

    def __init__(self):
        pass

    def draw_popup_fakec(self):
        with gr.Box(elem_id="fake-gradio-components", visible=False):
            self.updateChuanhuBtn = gr.Button(
                visible=False, elem_classes="invisible-btn", elem_id="update-chuanhu-btn")
            self.changeSingleSessionBtn = gr.Button(
                visible=False, elem_classes="invisible-btn", elem_id="change-single-session-btn")
            self.changeOnlineSearchBtn = gr.Button(
                visible=False, elem_classes="invisible-btn", elem_id="change-online-search-btn")
            self.historySelectBtn = gr.Button(
                visible=False, elem_classes="invisible-btn", elem_id="history-select-btn")  # Not used
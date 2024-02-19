var uploadInputElement = null

function move_cursor() {
    const buttonsParent = gradioApp().getElementById('prompt_list');
    if (buttonsParent && user_input_tb && user_input_ta) {
        buttonsParent.querySelectorAll('button').forEach((button) => {
            button.addEventListener('click', () => {
                user_input_ta.focus();
            });
        });
    }
}

function toast_move_main() {
    const spike_toast_info = gradioApp().getElementById('spike-toast-info');
    const gradio_main = gradioApp().querySelector('.gradio-container > .main')
    if (!gradio_main.querySelector('#spike-toast-info')) {
        gradio_main.appendChild(spike_toast_info);
    }
}

function check_move_list() {
    const promptList = document.getElementById('prompt_list');
    const pm_check = document.getElementById('pm_check');
    if (!promptList.querySelector('.label #pm_check')) {
        promptList.querySelector('.label').appendChild(pm_check);
    }
}

function btn_move_to_tab() {
    const mk_tabs = document.getElementById('mask_tabs');
    const mk_del = document.getElementById('mk_del');
    const mk_clear = document.getElementById('mk_clear');
    const mk_btn_wrap = mk_tabs.querySelector('.controls-wrap')
    if (!mk_btn_wrap.contains(mk_del)) {
        mk_btn_wrap.appendChild(mk_del)
    }
    if (!mk_btn_wrap.contains(mk_clear)) {
        mk_btn_wrap.appendChild(mk_clear)
    }
}

function red_dot_detection() {
    const langchainTab = document.getElementById('langchain_tab');
    const hasGeneratingAttr = langchainTab.querySelector('[generating]');
    if (hasGeneratingAttr) {
        const btn = document.getElementById("prompt-mask-btn")
        const dot = document.createElement('span');
        dot.className = 'red-dot';
        btn.appendChild(dot);
    }
}

function remove_red_dot() {
    const btn = document.getElementById("prompt-mask-btn");
    const redDot = btn.querySelector('.red-dot');
    if (redDot) {
        btn.removeChild(redDot);
    }
}

function reuse_or_edit(check, b, c, d, e, f) {
    if (check === false) {
        open_treasure_chest()
    }
    return [check, b, c, d, e, f]
}


function addShowAllButton() {
    var codeWraps = document.getElementsByClassName('code_wrap');
    for (var i = 0; i < codeWraps.length; i++) {
        if (codeWraps[i].getElementsByClassName('language-folded').length > 0) {
            // 检查button是否已经存在
            var existingButton = codeWraps[i].querySelector("#show-all-btn");
            // 如果按钮不存在，则创建
            if (!existingButton) {
                // 创建外层button元素
                var show_button = document.createElement("button");
                show_button.id = "show-all-btn";
                show_button.title = "show all";
                // 创建内部span包裹svg图标
                var span_show_svg = document.createElement("span");
                span_show_svg.innerHTML = `
                        <svg t="1699438091661" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg"
                            p-id="6097" width="17" height="17">
                            <path
                                d="M193.349046 210.094207v130.995519c0.084979 23.530755-6.042025 43.024996-29.555785 43.105726h-0.195452c-23.649726-0.025494-33.940714-19.53673-34.004448-43.207701v-169.916017a41.580349 41.580349 0 0 1 41.784299-41.809792h170.328166c23.641228 0.029743 42.795552 10.707386 42.825294 34.36561 0.029743 23.535004-19.009859 29.445311-42.523618 29.475054H210.344896a16.995851 16.995851 0 0 0-16.99585 16.99585z m620.306058-16.99585H681.992498c-23.513759-0.025494-42.549112-5.935801-42.523618-29.470805 0.029743-23.662473 19.184066-34.335867 42.825294-34.369859H852.78805a41.580349 41.580349 0 0 1 41.618589 41.809792v169.920266c-0.063734 23.666722-10.354722 43.182207-34.000199 43.21195h-0.199701c-23.513759-0.084979-29.636515-19.57922-29.555785-43.109975v-130.995519a16.995851 16.995851 0 0 0-16.99585-16.99585zM210.344896 830.09434H342.007502c23.513759 0.025494 42.553361 5.94005 42.523618 29.470805-0.029743 23.662473-19.184066 34.335867-42.825294 34.369859H171.21195a41.580349 41.580349 0 0 1-41.618589-41.809792v-169.916017c0.063734-23.670971 10.354722-43.186456 34.004448-43.21195h0.195452c23.513759 0.084979 29.636515 19.574971 29.555785 43.105726v130.995519a16.995851 16.995851 0 0 0 16.99585 16.99585z m620.306058-16.859884v-130.991269c-0.084979-23.535004 6.042025-43.024996 29.555785-43.109975h0.199701c23.645477 0.029743 33.936465 19.545228 34.000199 43.21195v169.916016a41.580349 41.580349 0 0 1-41.784299 41.809793h-170.328166c-23.641228-0.029743-42.795552-10.707386-42.825294-34.36561-0.025494-23.535004 19.009859-29.445311 42.523618-29.475054H813.655104a16.995851 16.995851 0 0 0 16.99585-16.995851z"
                                fill="currentColor" p-id="6098"></path>
                        </svg>
                `
                // 将span添加到button，并将button添加到父节点
                show_button.appendChild(span_show_svg);
                codeWraps[i].appendChild(show_button);
                codeWraps[i].querySelector('.language-folded').classList.add('clamp_2')
                // 给button添加监听事件
                show_button.addEventListener('click', handleShowAllButtonClick);
            }
        }
    }
}

function handleShowAllButtonClick(event) {
    if (!event.target.closest('#show-all-btn')) return; // 如果事件来源不在 Show All 按钮内执行退出

    var codeWrap = event.target.closest('.code_wrap');

    if (!codeWrap) { // 找不到外围code_wrap则退出运行
        console.warn("Can't find the parent .code_wrap element.");
        return;
    }
    var languageFoldedElement = codeWrap.querySelector('.language-folded');

    if (languageFoldedElement) {
        if (languageFoldedElement.classList.contains('unclamp')) {
            languageFoldedElement.classList.remove('unclamp');
            languageFoldedElement.classList.add('clamp_2')
        } else {
            languageFoldedElement.classList.add('unclamp');
            languageFoldedElement.classList.remove('clamp_2')
        }
    }
}


// 函数：当鼠标悬浮在 'uploaded-files-count' 或 'upload-index-file' 上时，改变 'upload-index-file' 的 display 样式为 flex
function showUploadIndexFile() {
    uploadIndexFileElement.style.display = "flow-root";
}


// 函数：当鼠标离开 'uploaded-files-count' 2秒 后，检查是否还处于 'upload-index-file' hover状态 ，如果否，则改变 'upload-index-file' 的 display样式 为 none
function hideUploadIndexFile() {
    setTimeout(function () {
        if (!isHover(uploadIndexFileElement)) {
            uploadIndexFileElement.style.display = "none";
        }
    }, 1000);
}

function isHover(e) {
    return (e.parentElement.querySelector(':hover') === e);
}

function add_func_event() {
    // 监听上传文件计数器元素和 upload-index-file 元素的 hover(in JS handle by mouseenter and mouseleave) 和 non-hover 事件
    uploadedFilesCountElement.addEventListener("mouseenter", showUploadIndexFile);
    uploadedFilesCountElement.addEventListener("mouseleave", hideUploadIndexFile);
    uploadIndexFileElement.addEventListener("mouseenter", showUploadIndexFile);
    uploadIndexFileElement.addEventListener("mouseleave", hideUploadIndexFile);
}

function add_func_paste(input) {
    let paste_files = [];
    if (input) {
        input.addEventListener("paste", async function (e) {
            const clipboardData = e.clipboardData || window.clipboardData;
            const items = clipboardData.items;
            if (items) {
                for (i = 0; i < items.length; i++) {
                    if (items[i].kind === "file") { // 确保是文件类型
                        const file = items[i].getAsFile();
                        // 将每一个粘贴的文件添加到files数组中
                        paste_files.push(file);
                        e.preventDefault();  // 避免粘贴文件名到输入框
                    }
                }
                if (paste_files.length > 0) {
                    // 按照文件列表执行批量上传逻辑
                    await paste_upload_files(paste_files);
                    paste_files = []

                }
            }
        });
    }
}

async function paste_upload_files(files) {
    uploadInputElement = uploadIndexFileElement.querySelector("input[type=file]");
    let totalSizeMb = 0
    if (files && files.length > 0) {
        // 执行具体的上传逻辑
        if (uploadInputElement) {
            for (let i = 0; i < files.length; i++) {
                // 将从文件数组中获取的文件大小(单位为字节)转换为MB，
                totalSizeMb += files[i].size / 1024 / 1024;
            }
            // 检查文件总大小是否超过20MB
            if (totalSizeMb > 20) {
                toast_push('⚠️文件夹大于20MB 🚀上传文件中', 2000)
                // return;  // 如果超过了指定大小, 可以不进行后续上传操作
            }
            // 监听change事件， 原生Gradio可以实现
            // uploadInputElement.addEventListener('change', function(){replace_input_string()});
            let event = new Event("change");
            Object.defineProperty(event, "target", {value: uploadInputElement, enumerable: true});
            Object.defineProperty(event, "currentTarget", {value: uploadInputElement, enumerable: true});
            Object.defineProperty(uploadInputElement, "files", {value: files, enumerable: true});
            uploadInputElement.dispatchEvent(event);
            // toast_push('🎉上传文件成功', 2000)
        } else {
            toast_push('请先清除上传文件区后，再执行上传', 1000)
        }
    }
}

function replace_input_string() {
    let attempts = 0;
    const maxAttempts = 50;  // 超时处理5秒～
    function findAndReplaceDownloads() {
        const filePreviewElement = uploadIndexFileElement.querySelector('.file-preview');
        if (filePreviewElement) {
            const downloadLinks = filePreviewElement.querySelectorAll('.download a');
            // Run the rest of your code only if links are found
            if (downloadLinks.length > 0) {
                const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif'];
                downloadLinks.forEach(function (downloadLink) {
                    let http_links = downloadLink.getAttribute('href')
                    let name_links = downloadLink.getAttribute('download')
                    let fileExtension = http_links.substring(http_links.lastIndexOf('.'));
                    if (imageExtensions.includes(fileExtension)) {
                        user_input_ta.value += `![${name_links}](${http_links})`;
                    } else {
                        user_input_ta.value += `[${name_links}](${http_links})`;
                    }
                    user_input_ta.style.height = 'auto';
                    user_input_ta.style.height = (user_input_ta.scrollHeight) + 'px';
                });
                clearInterval(manager);
            }
        }
        attempts++;
        if (attempts >= maxAttempts) {
            // Do something after max failed attempts.
            clearInterval(manager)
            console.log("Failed to find downloads");
        }
    }

    let manager = setInterval(findAndReplaceDownloads, 100);
}

//提示信息 封装
function toast_push(msg, duration) {
    duration = isNaN(duration) ? 3000 : duration;
    const m = document.createElement('div');
    m.innerHTML = msg;
    m.style.cssText = "font-size:  var(--text-md) !important; color: rgb(255, 255, 255);background-color: rgba(0, 0, 0, 0.6);padding: 10px 15px;margin: 0 0 0 -60px;border-radius: 4px;position: fixed;    top: 50%;left: 50%;width: 130px;text-align: center;";
    document.body.appendChild(m);
    setTimeout(function () {
        var d = 0.5;
        m.style.opacity = '0';
        setTimeout(function () {
            document.body.removeChild(m)
        }, d * 1000);
    }, duration);
}


function sm_move_more_label() {
    let more_label_group = chatbotArea.querySelector('#chatbot-input-more-area').querySelector('.chatbot-input-more-label-group');
    let more_sm_btn = chatbotArea.querySelector('#gr-chat-sm-column');
    let more_sm_select = chatbotArea.querySelector('#gr-know-sm-column');

    if (more_label_group && !more_label_group.contains(more_sm_btn) && !more_label_group.contains(more_sm_select)) {
        more_label_group.insertBefore(more_sm_select, more_label_group.firstChild);
        more_label_group.appendChild(more_sm_btn);
    }
}

var hintArea;

function setDragUploader() {
    input = chatbotArea;
    if (input) {
        const dragEvents = ["dragover", "dragenter"];
        const leaveEvents = ["dragleave", "dragend", "drop"];
        const onDrag = function (e) {
            e.preventDefault();
            e.stopPropagation();
            if (!chatbotArea.classList.contains("with-file")) {
                chatbotArea.classList.add("dragging");
                draggingHint();
            } else {
                toast_push('请先清除上传文件区后，再执行上传', 1000)
            }
        };

        const onLeave = function (e) {
            e.preventDefault();
            e.stopPropagation();
            chatbotArea.classList.remove("dragging");
            if (hintArea) {
                hintArea.remove();
            }
        };

        dragEvents.forEach(event => {
            input.addEventListener(event, onDrag);
        });

        leaveEvents.forEach(event => {
            input.addEventListener(event, onLeave);
        });

        input.addEventListener("drop", async function (e) {
            const files = e.dataTransfer.files;
            await paste_upload_files(files);
        });
    }
}

function draggingHint() {
    hintArea = chatbotArea.querySelector(".dragging-hint");
    if (hintArea) {
        return;
    }
    hintArea = document.createElement("div");
    hintArea.classList.add("dragging-hint");
    hintArea.innerHTML = `<div class="dragging-hint-text"><p>释放文件以上传</p></div>`;
    chatbotArea.appendChild(hintArea);
}


function insertFilePreview(fileRow) {
    if (fileRow) {
        // 判断是否已经添加过预览
        if (fileRow.getElementsByClassName('td-preview').length > 0) {
            return;
        }

        let tdElem = document.createElement("td");
        tdElem.className = "td-preview";  // 增加标识

        let link = fileRow.querySelector('.download a');
        if (!link) {
            return;
        }

        let extension = link.download.split('.').pop();

        if (validImgExtensions.includes(extension)) {     // 对于图片, 建立 <img>
            let img = document.createElement("img");
            img.src = link.href;
            img.className = 'td-a-preview';
            tdElem.appendChild(img);
        } else if (validDocsExtensions.includes(extension)) {  // 对于可读其他文件， 建立 <iframe>
            let iframe = document.createElement('iframe');
            iframe.src = link.href;
            iframe.className = 'td-a-preview';
            tdElem.appendChild(iframe);
        } else if (validAudioExtensions.includes(extension)) {   //对于音频文件，建立<audio>
            let audio = document.createElement('audio');
            audio.controls = true;    //增加控制条
            audio.src = link.href;
            audio.className = 'td-a-preview'
            tdElem.appendChild(audio);
        } else if (validVideoExtensions.includes(extension)) {   //对于视频文件，建立<video>
            let video = document.createElement('video');
            video.controls = true;      //增加控制条
            video.src = link.href;
            video.className = 'td-a-preview'
            tdElem.appendChild(video);
        } else {   // 对于不能在网页中预览的，增加一个提示
            let textNode = document.createTextNode("无法预览此类型的文件");
            let para = document.createElement('p');
            para.style.width = "100px";
            para.style.height = "75px";
            tdElem.appendChild(textNode);
        }

        fileRow.appendChild(tdElem);
    }
}

function addInputListeners() {
    for (const key in input_storage_mapping) {
        if (input_storage_mapping.hasOwnProperty(key)) {
            const inputElement = input_storage_mapping[key];
            inputElement.addEventListener('input', (function(key) {
                return function() {
                    localStorage.setItem(key, this.value);
                };
            })(key));
        }
    }
}

function fillInputsFromCache() {
    for (const key in input_storage_mapping) {
        if (input_storage_mapping.hasOwnProperty(key)) {
            const inputElement = input_storage_mapping[key];
            const cachedValue = localStorage.getItem(key);
            if (cachedValue && cachedValue !== 'undefined') {
                inputElement.value = cachedValue;
            }
        }
    }
}


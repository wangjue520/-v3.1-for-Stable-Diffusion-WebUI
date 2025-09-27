import gradio as gr
import modules.scripts as scripts
import os
import json
import sys
import subprocess
import importlib
from datetime import datetime
import io

# --- 依赖检查与自动安装 (使用opencv-contrib-python) ---
REQUIRED_PACKAGES = {"numpy": "numpy", "PIL": "Pillow", "cv2": "opencv-contrib-python"}
LIBRARIES_LOADED = True
RESTART_REQUIRED = False
ERROR_MESSAGE = ""
for import_name, package_name in REQUIRED_PACKAGES.items():
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"\n!!! [图像后期处理脚本] 模块 '{import_name}' 未找到，正在尝试自动安装...")
        try:
            if package_name == "opencv-contrib-python":
                subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'opencv-python'])
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name, '--upgrade'])
            print(f"--- ✓ '{package_name}' 安装/升级成功! 请完全重启 Stable Diffusion WebUI 以使脚本生效。")
            RESTART_REQUIRED = True
            LIBRARIES_LOADED = False
        except Exception as e:
            LIBRARIES_LOADED = False
            ERROR_MESSAGE = (f"自动安装 '{package_name}' 失败！<br>" f"错误: {e}<br><br>" "<b>请尝试手动安装: pip install opencv-contrib-python --upgrade</b>")
            print(f"!!! ✗ 自动安装 '{package_name}' 失败! 错误: {e}")
            break
if LIBRARIES_LOADED:
    import numpy as np
    from PIL import Image, ImageColor
    import cv2

# --- 全局设置 ---
SCRIPT_DIR = os.path.dirname(__file__)
SAVE_DIR = os.path.join(SCRIPT_DIR, "post_processing_profiles_v3")
STANDALONE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "standalone_output")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(STANDALONE_OUTPUT_DIR, exist_ok=True)
CONFIG_VERSION = 3

class Script(scripts.Script):
    def title(self):
        return "图像后期处理 (v3)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    # --- 图像处理算法 ---
    def pil_to_cv2(self, pil_image):
        if pil_image is None: return None
        if pil_image.mode == 'RGBA': return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)
        else: return cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)

    def cv2_to_pil(self, cv2_image):
        if cv2_image is None: return None
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

    def apply_noise(self, img, noise_type, amount):
        if amount == 0 or noise_type == '无': return img
        if noise_type == '椒盐':
            h, w, _ = img.shape; num_pixels = h * w; num_noise_pixels = int(num_pixels * amount)
            s_vs_p = 0.5; num_salt = int(num_noise_pixels * s_vs_p); num_pepper = num_noise_pixels - num_salt
            salt_coords_y = np.random.randint(0, h, num_salt); salt_coords_x = np.random.randint(0, w, num_salt)
            img[salt_coords_y, salt_coords_x] = 255
            pepper_coords_y = np.random.randint(0, h, num_pepper); pepper_coords_x = np.random.randint(0, w, num_pepper)
            img[pepper_coords_y, pepper_coords_x] = 0
            return img
        h, w, c = img.shape; noise = np.zeros((h,w,c), np.int16)
        if noise_type == '高斯': cv2.randn(noise, 0, amount * 128)
        elif noise_type == '均匀': cv2.randu(noise, -amount * 128, amount * 128)
        elif noise_type == '斑点':
            gauss = np.random.normal(0, amount, img.shape); noisy = img/255.0 + gauss
            return np.clip(noisy * 255, 0, 255).astype(np.uint8)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def apply_film_grain(self, img, amount):
        if amount == 0: return img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); noise = np.zeros(gray.shape, np.int16)
        cv2.randn(noise, 0, amount * 50)
        noisy_gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); l, a, b = cv2.split(lab)
        merged_lab = cv2.merge([noisy_gray, a, b])
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    def apply_blur(self, img, blur_type, kernel_size, motion_angle=0):
        if kernel_size == 0 or blur_type == '无': return img
        k = kernel_size * 2 + 1
        if blur_type == '高斯': return cv2.GaussianBlur(img, (k, k), 0)
        elif blur_type == '运动':
            kernel = np.zeros((k, k)); angle = motion_angle * np.pi / 180.
            x, y = np.cos(angle), np.sin(angle); center = k // 2
            cv2.line(kernel, (int(center-x*center), int(center-y*center)), (int(center+x*center), int(center+y*center)), 1.0)
            kernel /= np.sum(kernel)
            return cv2.filter2D(img, -1, kernel)
        return img
    
    def apply_stylize(self, img, style_type, p1, p2):
        if style_type == '无': return img
        if style_type == '素描/铅笔画':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); inv = 255 - gray
            k = int(p1) * 2 + 1; blur = cv2.GaussianBlur(inv, (k, k), 0)
            sketch = cv2.divide(gray, 255 - blur, scale=256.0)
            return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        if style_type == '边缘检测':
            t1, t2 = int(p1), int(p2); edges = cv2.Canny(img, t1, t2)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        if style_type == '怀旧棕调':
            kernel = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
            return np.clip(cv2.transform(img, kernel), 0, 255).astype(np.uint8)
        if style_type == '反色': return 255 - img
        if style_type == '像素化':
            block_size = int(p1);
            if block_size < 2: return img
            h, w, _ = img.shape
            temp = cv2.resize(img, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        if style_type == '雨滴':
            density = p1 / 100.0; 
            if density == 0: return img
            h, w, _ = img.shape; overlay = img.copy()
            num_drops = int(h * w * density * 0.01)
            for _ in range(num_drops):
                x, y = np.random.randint(0, w), np.random.randint(0, h)
                length, angle = np.random.randint(5, 30), np.random.randint(75, 105)
                cv2.line(overlay, (x, y), (x + int(length*np.cos(angle*np.pi/180)), y + int(length*np.sin(angle*np.pi/180))), (230,230,230), 1)
            return cv2.addWeighted(img, 0.8, overlay, 0.2, 0)
        return img
    
    def apply_monochrome(self, img, dark_color_hex, light_color_hex, intensity):
        if intensity == 0: return img
        dark_bgr = np.array(ImageColor.getrgb(dark_color_hex)[::-1], dtype=np.float32)
        light_bgr = np.array(ImageColor.getrgb(light_color_hex)[::-1], dtype=np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32) / 255.0
        duotone_f = dark_bgr * (1 - gray_f[..., np.newaxis]) + light_bgr * gray_f[..., np.newaxis]
        duotone = np.clip(duotone_f, 0, 255).astype(np.uint8)
        return cv2.addWeighted(img, 1 - intensity, duotone, intensity, 0)

    def apply_all_adjustments(self, pil_image, *args):
        if not LIBRARIES_LOADED or pil_image is None: 
            print("[Post-Processing] 错误：依赖库未加载或输入图片为空")
            return None
        params_dict = self.args_to_dict(args)
        
        img_np_f = np.array(pil_image.convert('RGB'), dtype=np.float32) / 255.0
        
        img_np_f *= (2 ** params_dict['exposure'])
        img_np_f = 0.5 + (img_np_f - 0.5) * params_dict['contrast']
        if params_dict['temperature'] != 0:
            img_np_f[:, :, 0] += params_dict['temperature']; img_np_f[:, :, 2] -= params_dict['temperature']
        if params_dict['tint'] != 0: img_np_f[:, :, 1] += params_dict['tint']
        if params_dict['saturation'] != 1.0 or params_dict['vibrance'] != 0:
            # 先确保值在0-1范围内
            img_to_convert = np.clip(img_np_f, 0, 1)
            # hsv = cv2.cvtColor((img_np_f*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv = cv2.cvtColor((img_to_convert*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            
            hsv[:,:,1] *= params_dict['saturation']
            if params_dict['vibrance'] != 0:
                #luma = (img_np_f * np.array([0.299, 0.587, 0.114])).sum(axis=-1)
                luma = (img_to_convert * np.array([0.299, 0.587, 0.114])).sum(axis=-1)
                sat_mult = np.abs(luma - 0.5) * 2
                hsv[:,:,1] += params_dict['vibrance'] * (1 - hsv[:,:,1]/255.0) * sat_mult * 128
            hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
            img_np_f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)/255.0
        if params_dict['highlights'] != 0 or params_dict['shadows_adj'] != 0:
            lum = (img_np_f * np.array([0.299, 0.587, 0.114])).sum(axis=-1)
            sm, hm = (1-lum)**2, lum**2
            if params_dict['shadows_adj']!=0: img_np_f += params_dict['shadows_adj'] * sm[...,np.newaxis]
            if params_dict['highlights']!=0: img_np_f += params_dict['highlights'] * hm[...,np.newaxis]
        img_np_f = img_np_f * (1 + params_dict['whites']) - params_dict['blacks']
        img_np_f = np.clip(img_np_f, 0, 1)

        cv2_img = (img_np_f * 255).astype(np.uint8)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        
        cv2_img = self.apply_monochrome(cv2_img, params_dict['mono_dark'], params_dict['mono_light'], params_dict['mono_intensity'])
        if params_dict['sharpen_amount'] > 0:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharp = cv2.filter2D(cv2_img, -1, kernel)
            cv2_img = cv2.addWeighted(cv2_img, 1-params_dict['sharpen_amount'], sharp, params_dict['sharpen_amount'], 0)
        cv2_img = self.apply_blur(cv2_img, params_dict['blur_type'], params_dict['blur_kernel'], params_dict['motion_angle'])
        cv2_img = self.apply_stylize(cv2_img, params_dict['style_type'], params_dict['style_p1'], params_dict['style_p2'])
        cv2_img = self.apply_noise(cv2_img, params_dict['noise_type'], params_dict['noise_amount'])
        cv2_img = self.apply_film_grain(cv2_img, params_dict['grain_amount'])
        
        processed = self.cv2_to_pil(cv2_img)
        if processed is None:
            print("[Post-Processing] 错误：后处理图片生成失败")
        else:
            print("[Post-Processing] 成功生成处理后的图片")
        return processed

    def get_component_keys(self):
        return [
            "exposure", "contrast", "highlights", "shadows_adj", "whites", "blacks", "saturation", "vibrance", "temperature", "tint",
            "mono_dark", "mono_light", "mono_intensity", "s_color", "m_color", "h_color", "s_intensity", "m_intensity", "h_intensity",
            "blur_type", "blur_kernel", "motion_angle", "sharpen_amount", "noise_type", "noise_amount", "grain_amount",
            "style_type", "style_p1", "style_p2"
        ]

    def args_to_dict(self, args):
        return dict(zip(self.get_component_keys(), args))

    def create_ui_components(self):
        with gr.Tab("基础调整"):
            exposure = gr.Slider(-1.0, 1.0, 0.0, step=0.05, label="曝光")
            contrast = gr.Slider(0.5, 2.0, 1.0, step=0.05, label="对比度")
            highlights = gr.Slider(-1.0, 1.0, 0.0, step=0.05, label="高光")
            shadows_adj = gr.Slider(-1.0, 1.0, 0.0, step=0.05, label="阴影")
            whites = gr.Slider(-0.5, 0.5, 0.0, step=0.01, label="白色色阶")
            blacks = gr.Slider(-0.5, 0.5, 0.0, step=0.01, label="黑色色阶")
            saturation = gr.Slider(0.0, 2.0, 1.0, step=0.05, label="饱和度")
            vibrance = gr.Slider(-1.0, 1.0, 0.0, step=0.05, label="自然饱和度")
            temperature = gr.Slider(-0.2, 0.2, 0.0, step=0.005, label="色温")
            tint = gr.Slider(-0.2, 0.2, 0.0, step=0.005, label="色调")
        with gr.Tab("单色 & 渐变"):
            mono_intensity = gr.Slider(0, 1, 0, step=0.05, label="强度")
            with gr.Row():
                mono_dark = gr.ColorPicker("#000000", label="暗部颜色")
                mono_light = gr.ColorPicker("#FFFFFF", label="亮部颜色")
        with gr.Tab("高级色彩分级"):
            s_color = gr.ColorPicker("#0000FF", label="阴影色调"); m_color = gr.ColorPicker("#FFFFFF", label="中间调色调"); h_color = gr.ColorPicker("#FFFF00", label="高光色调")
            s_intensity = gr.Slider(0,1,0, step=0.01, label="阴影强度"); m_intensity = gr.Slider(0,1,0, step=0.01, label="中间调强度"); h_intensity = gr.Slider(0,1,0, step=0.01, label="高光强度")
        with gr.Tab("模糊 & 锐化"):
            blur_type = gr.Radio(["无", "高斯", "运动"], value="无", label="模糊类型"); blur_kernel = gr.Slider(0, 50, 0, step=1, label="模糊半径")
            motion_angle = gr.Slider(0, 180, 0, step=1, label="运动模糊角度"); sharpen_amount = gr.Slider(0, 1, 0, step=0.05, label="锐化强度")
        with gr.Tab("噪点 & 颗粒"):
            noise_type = gr.Radio(["无", "高斯", "椒盐", "均匀", "斑点"], value="无", label="噪点类型"); noise_amount = gr.Slider(0, 1, 0, step=0.01, label="噪点强度")
            grain_amount = gr.Slider(0, 1, 0, step=0.01, label="胶片颗粒")
        with gr.Tab("风格化滤镜"):
            style_type = gr.Radio(["无", "素描/铅笔画", "边缘检测", "怀旧棕调", "反色", "像素化", "雨滴"], value="无", label="滤镜类型 (互斥)")
            with gr.Row():
                style_p1 = gr.Slider(1, 255, 10, step=1, label="参数1"); style_p2 = gr.Slider(1, 255, 100, step=1, label="参数2")
            gr.Markdown("参数说明: 素描(模糊核), 边缘检测(阈值1, 阈值2), 像素化(块大小), 雨滴(密度/100)")
        return [
            exposure, contrast, highlights, shadows_adj, whites, blacks, saturation, vibrance, temperature, tint,
            mono_dark, mono_light, mono_intensity, s_color, m_color, h_color, s_intensity, m_intensity, h_intensity,
            blur_type, blur_kernel, motion_angle, sharpen_amount, noise_type, noise_amount, grain_amount,
            style_type, style_p1, style_p2
        ]

    def create_standalone_ui(self):
        with gr.Column(visible=False) as ui_content:
            gr.Markdown("## 🎨 独立图像调色工具")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 1. 上传图片"); sa_input_image = gr.Image(type="pil", label="上传或拖入图片")
                    self.sa_components = self.create_ui_components()
                    with gr.Accordion("💾 保存与加载预设", open=False):
                        sa_save_name = gr.Textbox(label="保存预设", value="preset"); sa_save_btn = gr.Button("保存", variant='primary')
                        sa_load_dropdown = gr.Dropdown(choices=self.get_json_list(), label="加载预设"); sa_load_btn = gr.Button("加载")
                with gr.Column(scale=1):
                    gr.Markdown("#### 2. 实时预览与保存"); sa_output_image = gr.Image(type="pil", label="效果预览", interactive=False)
                    sa_save_image_btn = gr.Button("💾 保存图片", variant='primary'); sa_download_file = gr.File(label="下载", visible=False)
                    close_btn = gr.Button("收起独立工具")
        all_sa_inputs = [sa_input_image] + self.sa_components
        for component in all_sa_inputs:
            component.change(self.apply_all_adjustments, inputs=all_sa_inputs, outputs=[sa_output_image])
        sa_save_btn.click(self.save_preset, inputs=[sa_save_name] + self.sa_components, outputs=[sa_load_dropdown])
        sa_load_btn.click(self.load_preset, inputs=[sa_load_dropdown], outputs=self.sa_components)
        sa_save_image_btn.click(self.save_standalone_image, inputs=[sa_output_image], outputs=[sa_download_file])
        return ui_content, close_btn

    def ui(self, is_img2img):
        if not LIBRARIES_LOADED:
            msg = f"<div style='padding:20px;border:2px solid #dc3545;border-radius:5px;background-color:#f8d7da;'><h3 style='color:#721c24;'>错误：脚本加载失败</h3><p style='color:#721c24;'>{ERROR_MESSAGE}</p></div>"
            if RESTART_REQUIRED: msg = "<div style='padding:20px;border:2px solid #17a2b8;border-radius:5px;background-color:#d1ecf1;'><h3 style='color:#0c5460;'>提示：需要重启</h3><p style='color:#0c5460;'>脚本依赖库已成功安装/升级！</p><p style='color:#0c5460;'>请<b>完全重启 Stable Diffusion WebUI</b>以加载并使用此脚本。</p></div>"
            with gr.Accordion(self.title(), open=True): gr.HTML(msg)
            return []

        gr.HTML("<style>#launch_standalone_button { background: linear-gradient(45deg, #FF8C00, #FFA500); color: white; border: none; } </style>")
        launch_button = gr.Button("🚀 启动独立调色工具", elem_id="launch_standalone_button")
        standalone_ui_container, close_button = self.create_standalone_ui()
        launch_button.click(lambda: gr.update(visible=True), outputs=[standalone_ui_container])
        close_button.click(lambda: gr.update(visible=False), outputs=[standalone_ui_container])

        with gr.Accordion("🎨 图像后期处理 (集成流水线)", open=True, elem_id="postprocessing_integrated_accordion"):
            with gr.Row():
                enable_post_processing = gr.Checkbox(label="启用图像后处理", value=False)
                save_original = gr.Checkbox(label="额外保存原图", value=False)
            info_suffix = gr.Textbox(label="图片信息后缀", value="-post", max_lines=1)
            main_ui_components = self.create_ui_components()
            self.integrated_components = [enable_post_processing, save_original, info_suffix] + main_ui_components
        with gr.Accordion("💾 保存与加载预设 (集成流水线)", open=False):
            save_name = gr.Textbox(label="保存预设", value="default_preset"); save_btn = gr.Button("保存", variant='primary')
            load_dropdown = gr.Dropdown(choices=self.get_json_list(), label="加载预设"); load_btn = gr.Button("加载")
        
        preset_components = self.integrated_components[3:]
        save_btn.click(self.save_preset, inputs=[save_name] + preset_components, outputs=[load_dropdown])
        load_btn.click(self.load_preset, inputs=[load_dropdown], outputs=preset_components)
        
        return self.integrated_components

    def get_json_list(self): 
        if not LIBRARIES_LOADED: return []
        return [f[:-5] for f in os.listdir(SAVE_DIR) if f.endswith(".json")]

    def save_standalone_image(self, image):
        if image is None: return gr.update(visible=False)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S"); filename = f"standalone_{timestamp}.png"
        path = os.path.join(STANDALONE_OUTPUT_DIR, filename)
        image.save(path); return gr.File.update(value=path, visible=True)

    def save_preset(self, filename, *args):
        data = {"config_version": CONFIG_VERSION, "settings": {}}
        data["settings"] = self.args_to_dict(args)
        if not filename.endswith(".json"): filename += ".json"
        path = os.path.join(SAVE_DIR, filename)
        with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=4, ensure_ascii=False)
        return gr.Dropdown.update(choices=self.get_json_list(), value=filename.replace('.json', ''))

    def load_preset(self, filename):
        if not filename: return [0.0] * 29
        path = os.path.join(SAVE_DIR, f"{filename}.json")
        if not os.path.exists(path): raise FileNotFoundError(f"预设文件 {filename}.json 未找到！")
        with open(path, "r", encoding="utf-8") as f: data = json.load(f)
        settings = data.get("settings", {})
        defaults = { "exposure": 0.0, "contrast": 1.0, "highlights": 0.0, "shadows_adj": 0.0, "whites": 0.0, "blacks": 0.0, "saturation": 1.0, "vibrance": 0.0, "temperature": 0.0, "tint": 0.0, "mono_dark": "#000000", "mono_light": "#FFFFFF", "mono_intensity": 0.0, "s_color": "#0000FF", "m_color": "#FFFFFF", "h_color": "#FFFF00", "s_intensity": 0.0, "m_intensity": 0.0, "h_intensity": 0.0, "blur_type": "无", "blur_kernel": 0, "motion_angle": 0, "sharpen_amount": 0, "noise_type": "无", "noise_amount": 0, "grain_amount": 0, "style_type": "无", "style_p1": 10, "style_p2": 100 }
        return [settings.get(key, defaults.get(key)) for key in self.get_component_keys()]

    def postprocess_image(self, p, pp, *args):
        is_enabled, save_original_flag, suffix = args[0], args[1], args[2]
        params = args[3:]
        if not is_enabled or not LIBRARIES_LOADED:
            print("[Post-Processing] 未启用后处理或依赖库未加载，跳过后处理")
            return

        print(f"[Post-Processing] v3 开始处理图片，启用后处理：{is_enabled}，保存原图：{save_original_flag}，后缀：{suffix}")

        # 存储原图
        original_image = pp.image.copy() if pp.image else None
        if original_image is None:
            print("[Post-Processing] 错误：输入图片为空。")
            return

        # 执行后处理
        processed_image = self.apply_all_adjustments(pp.image, *params)
        if processed_image is None:
            print("[Post-Processing] 错误：后处理失败，未生成处理后图片。")
            return

        # 将处理后的图片设置为主要输出
        pp.image = processed_image

        # **核心更改：** 如果启用了“额外保存原图”，则手动保存原始图片
        if save_original_flag and original_image:
            print("[Post-Processing] 额外保存原图，将原始图片手动保存。")
            
            # 使用一个更可靠的方法来获取基础路径
            try:
                # 尝试从 p 对象中获取路径，这是最理想的情况
                base_dir = p.outpath_txt2img if hasattr(p, 'outpath_txt2img') else p.outpath
            except AttributeError:
                # 如果 p 对象没有这些属性，则使用通用的输出路径
                try:
                    import modules.shared as shared
                    base_dir = shared.opts.outdir_txt2img_samples
                except ImportError:
                    # 最终的退路：使用脚本所在的目录
                    base_dir = os.path.join(scripts.basedir(), "outputs", "post-processing-extras")

            # 构造手动保存路径
            # 创建一个子文件夹来保存额外的图片
            extra_dir = os.path.join(base_dir, "post-processing-extras")
            os.makedirs(extra_dir, exist_ok=True)
            
            # 使用时间戳作为文件名
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            original_filename = f"{timestamp}-original.png"
            original_path = os.path.join(extra_dir, original_filename)
            
            try:
                original_image.save(original_path)
                print(f"[Post-Processing] 原始图片已手动保存到: {original_path}")
            except Exception as e:
                print(f"[Post-Processing] 错误：手动保存原始图片失败！{e}")
        
        # 添加后处理参数到元数据
        params_dict = self.args_to_dict(params)
        defaults = {
            "exposure": 0.0, "contrast": 1.0, "highlights": 0.0, "shadows_adj": 0.0,
            "whites": 0.0, "blacks": 0.0, "saturation": 1.0, "vibrance": 0.0,
            "temperature": 0.0, "tint": 0.0, "mono_dark": "#000000",
            "mono_light": "#FFFFFF", "mono_intensity": 0.0, "s_color": "#0000FF",
            "m_color": "#FFFFFF", "h_color": "#FFFF00", "s_intensity": 0.0,
            "m_intensity": 0.0, "h_intensity": 0.0, "blur_type": "无",
            "blur_kernel": 0, "motion_angle": 0, "sharpen_amount": 0,
            "noise_type": "无", "noise_amount": 0, "grain_amount": 0,
            "style_type": "无", "style_p1": 10, "style_p2": 100
        }
        active_params = {k: v for k, v in params_dict.items() if v != defaults.get(k)}
        if active_params:
            info_text = f"PostProcess: {suffix}, Params: {json.dumps(active_params, ensure_ascii=False)}"
            p.extra_generation_params["Post-Process"] = info_text
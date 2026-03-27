import pandas as pd
import json
import requests
import logging
import time
import sys
from typing import Dict, Any, List, Optional
from openai import OpenAI
from PIL import Image
from transformers import (Blip2Processor, Blip2ForConditionalGeneration, 
                         AutoTokenizer, AutoModelForSequenceClassification)
from config import LLMConfig, load_config
import torch
import torch.nn as nn
from torchvision import models, transforms
import easyocr
import os
import glob
import face_recognition

# 根据你的CPU核心数调整
torch.set_num_threads(4)  
os.environ['OMP_NUM_THREADS'] = '4'

# 全局情感预测器实例
predictor = None



class CNNFeatureExtractor(nn.Module):
    """CNN 替代 ViT 提取视觉特征"""
    def __init__(self, output_dim=768):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2] 
        self.backbone = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, output_dim)

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)  # [B, 2048, H, W]
            x = self.pool(x).squeeze(-1).squeeze(-1)  # [B, 2048]
            x = self.fc(x)  # [B, output_dim]
        return x


class CNNBLIP2Analyzer:
    """CNN + BLIP2 图文识别模块"""
    def __init__(self, device='cpu'):
        self.device = device

        # 加载 CNN 提取器
        self.cnn_extractor = CNNFeatureExtractor(output_dim=768).to(device).eval()

        # 加载 BLIP2 模型
        print("🔧 正在加载 BLIP2 模型（使用 CPU）...")
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        print("BLIP2模型加载完成")

        # 替换 vision encoder 获取特征的逻辑
        self.model.get_vision_features = self.get_vision_features

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def get_vision_features(self, image: Image.Image):
        """重写 BLIP 的视觉特征提取逻辑（使用 CNN）"""
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        features = self.cnn_extractor(image_tensor)  # shape: [1, 768]
        return features

    def generate_caption(self, image_path: str, prompt="a photo of"):
        """给图片生成说明性文字"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=30)
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return caption
# 设置为评估模式，提高推理速度
print("BLIP2模型加载完成")

def preprocess_image(image_path: str) -> Image.Image:
    """预处理图片，确保格式和尺寸适合CPU处理"""
    try:
        image = Image.open(image_path)
        # 转换为RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # CPU处理时限制更小的尺寸以提高速度
        max_size = 512  # CPU模式下使用更小尺寸
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        raise Exception(f"图片预处理失败: {e}")

def load_known_faces(directory="utils/known_faces"):
    """从图库中加载需要识别的人脸图像"""
    try:
        known_encodings = []
        known_names = []
        for file in os.listdir(directory):
            path = os.path.join(directory, file)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(file)[0])
        print("图库加载完成")
        return known_encodings, known_names
    except Exception as e:
        raise Exception(f"加载图库失败: {e}")

known_encodings, known_names = load_known_faces()

def match_known_person(image_path):
    """输入图像与图库匹配，确定身份"""
    unknown_image = face_recognition.load_image_file(image_path)
    unknown_encodings = face_recognition.face_encodings(unknown_image)
    if not unknown_encodings:
        return "未检测到人脸"
    for unknown_encoding in unknown_encodings:
        results = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=0.5)
        if True in results:
            return known_names[results.index(True)]
    return "未识别人物"

class SentimentPredictor:
    def __init__(self, model_path):
        """
        初始化情感预测器
        Args:
            model_path: 训练好的模型路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.max_length = 128
        
        # 标签映射
        self.id_to_label = {0: '负向', 1: '中性', 2: '正向'}
        
        # 加载模型和tokenizer
        self.load_model()
        
    def load_model(self):
        """加载训练好的模型"""
        try:
            print(f"正在从 {self.model_path} 加载情感分析模型...")
            
            # 检查模型文件是否存在
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
                
            # 加载tokenizer和模型
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"情感分析模型加载成功！使用设备: {self.device}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)
            
    def predict_single_text(self, text):
        """
        预测单条文本的情感
        Args:
            text: 待预测的文本
        Returns:
            预测的情感标签和置信度
        """
        if not text or pd.isna(text) or str(text).strip() == "":
            return "中性", 1.0
            
        # 文本预处理
        text = str(text).strip()
        
        # 分词和编码
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence = predictions.max().item()
            predicted_class = torch.argmax(predictions, dim=-1).item()
            
        # 返回情感标签和置信度
        return self.id_to_label[predicted_class], round(confidence, 4)

def analyze_ev_image(image_path: str) -> str:
    """
    分析新能源汽车相关的图片内容：
    - 用 CNN+BLIP 提取图像内容描述
    - 用 EasyOCR 提取图像中文字内容
    - 人脸识别匹配
    - BERT模型分析文字情感
    
    输入: 本地图片路径
    返回: 综合描述字符串
    """
    global predictor
    
    try:
        # 初始化CNN-BLIP2分析器
        global analyzer
        if 'analyzer' not in globals():
            print("正在加载分析器...")
            analyzer = CNNBLIP2Analyzer(device='cpu')
            print("分析器加载完成")
        
        # 1. 图像内容识别 (CNN + BLIP2)
        blip_result = analyzer.generate_caption(image_path)
        
        # 2. 图像文字识别 (EasyOCR) 
        reader = easyocr.Reader(['ch_sim', 'en'], gpu=False) 
        ocr_result = reader.readtext(image_path, detail=0)
        ocr_text = ' '.join(ocr_result) if ocr_result else "无可识别文字"
        
        # 3. 文字情感分析 (BERT)
        if ocr_text == "无可识别文字":
            ocr_text_bert = None
        else:
            ocr_text_bert = ocr_text
        bert_text = ocr_text_bert + blip_result  # ocr部分和blip识别内容拼接
        sentiment_label = "中性"
        confidence = 1.0
        
        if bert_text.strip() and predictor is not None:
            try:
                sentiment_label, confidence = predictor.predict_single_text(bert_text)
            except Exception as e:
                logger.warning(f"情感分析失败: {e}")
                sentiment_label = "中性"
                confidence = 0.0
        
        # 4. 人脸识别 
        scanner = match_known_person(image_path)
        
        # 5. 合并输出 
        combined_description = (
            f"[图像内容识别]: {blip_result}\n"
            f"[图像文字识别]: {ocr_text}\n"
            f"[内容情感分析]: {sentiment_label} (置信度: {confidence:.4f})\n"
            f"[人脸匹配识别]: {scanner}"
        )
        
        return combined_description
        
    except Exception as e:
        logger.error(f"图片分析失败: {e}")
        return f"图片分析失败: {str(e)}"

def analyze_multiple_images(path: str) -> str:
    """
    分析多张图片或指定文件夹中的所有图片（CPU优化版本）：
    - 支持单个图片路径
    - 支持文件夹路径（分析文件夹内所有图片）
    
    输入: 图片路径或文件夹路径
    返回: 所有图片的综合分析结果
    """
    global predictor
    
    # 支持的图片格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_paths = []
    
    # 判断输入是文件还是文件夹
    if os.path.isfile(path):
        # 单个文件
        if any(path.lower().endswith(fmt) for fmt in supported_formats):
            image_paths = [path]
        else:
            return f"错误: 文件 {path} 不是支持的图片格式"
    elif os.path.isdir(path):
        # 文件夹 - 查找所有图片文件
        for fmt in supported_formats:
            # 使用glob查找并统一转换为小写路径去重
            lower_paths = [p.lower() for p in glob.glob(os.path.join(path, f"*{fmt}"))]
            upper_paths = [p.lower() for p in glob.glob(os.path.join(path, f"*{fmt.upper()}"))]
            all_paths = lower_paths + upper_paths
            
            unique_paths = []
            seen = set()
            for p in all_paths:
                if p not in seen:
                    seen.add(p)
                    original_path = next((op for op in glob.glob(os.path.join(path, f"*{fmt}")) + 
                                         glob.glob(os.path.join(path, f"*{fmt.upper()}")) 
                                         if op.lower() == p), None)
                    if original_path:
                        unique_paths.append(original_path)
            
            image_paths.extend(unique_paths)
        
        image_paths = list(set(image_paths))
        
        if not image_paths:
            return f"错误: 文件夹 {path} 中未找到支持的图片文件"
    else:
        return f"错误: 路径 {path} 不存在"
    
    max_images = 10  # 批量处理数量
    if len(image_paths) > max_images:
        image_paths = image_paths[:max_images]
        logger.warning(f"CPU模式下图片数量超过{max_images}张，仅处理前{max_images}张")
    
    # 分析每张图片
    results = []
    
    # 初始化EasyOCR
    try:
        reader = easyocr.Reader(['ch_sim', 'en'], gpu=False) 
        logger.info("EasyOCR初始化成功（CPU模式）")
    except Exception as e:
        logger.warning(f"EasyOCR初始化失败: {e}")
        reader = None
    
    for i, img_path in enumerate(image_paths, 1):
        try:
            logger.info(f"正在分析第 {i}/{len(image_paths)} 张图片: {os.path.basename(img_path)} (CPU模式，请耐心等待...)")
            
            try:
                global analyzer
                if 'analyzer' not in globals():
                    analyzer = CNNBLIP2Analyzer(device='cpu')
                
                blip_result = analyzer.generate_caption(img_path)
            except Exception as e:
                logger.warning(f"CNN-BLIP分析器初始化失败: {e}")
                analyzer = None
            
            # 图像文字识别 (EasyOCR)
            try:
                if reader is not None:
                    ocr_result = reader.readtext(img_path, detail=0)
                    ocr_text = ' '.join(ocr_result) if ocr_result else "无可识别文字"
                else:
                    ocr_text = "OCR功能不可用"
            except Exception as e:
                logger.error(f"OCR处理图片 {img_path} 时出错: {e}")
                ocr_text = f"文字识别失败: {str(e)}"
            
            # 文字情感分析 (BERT) 
            sentiment_label = "中性"
            confidence = 1.0
            
            if ocr_text.strip() and predictor is not None:
                try:
                    sentiment_label, confidence = predictor.predict_single_text(ocr_text)
                except Exception as e:
                    logger.warning(f"图片 {img_path} 情感分析失败: {e}")
                    sentiment_label = "中性"
                    confidence = 0.0
            
            # 人脸识别
            try:
                scanner = match_known_person(img_path)
            except Exception as e:
                logger.error(f"人脸识别处理图片 {img_path} 时出错: {e}")
                scanner = "人脸识别失败"
            
            # 单张图片结果
            img_result = (f"文件: {os.path.basename(img_path)}\n"
                         f"内容: {blip_result}\n"
                         f"文字: {ocr_text}\n"
                         f"文字情感: {sentiment_label} (置信度: {confidence:.4f})\n"
                         f"人脸: {scanner}")
            results.append(img_result)
            
        except Exception as e:
            logger.error(f"分析图片 {img_path} 时出错: {e}")
            results.append(f"文件: {os.path.basename(img_path)}\n错误: 分析失败 - {str(e)}")
    
    # 汇总结果
    summary = f"共分析了 {len(image_paths)} 张图片 (CPU模式):\n\n" + "\n\n---\n\n".join(results)
    return summary

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AVAILABLE_FUNCTIONS = {
    "analyze_ev_image": analyze_ev_image,
    "analyze_multiple_images": analyze_multiple_images,
}

FUNCTION_DESCRIPTIONS = [
    {
        "name": "analyze_ev_image",
        "description": "分析一张新能源车企相关的图片，例如车辆外观、创始人照片或品牌logo",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "本地图片文件路径，如 '/Users/xxx/Desktop/ev.jpg'"
                }
            },
            "required": ["image_path"]
        }
    },
    {
        "name": "analyze_multiple_images",
        "description": "批量分析多张图片或指定文件夹中的所有图片，生成综合报告",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "图片路径或文件夹路径，如 '/Users/xxx/Desktop/images/' 或单个图片路径"
                }
            },
            "required": ["path"]
        }
    }
]

#LLM适配器，支持DeepSeek和千问
class LLMAdapter:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.current_api = "auto"  # 'deepseek', 'qwen', 'blip2' or 'auto'
        self.deepseek_client = None
        self.deepseek_available = False
        self.qwen_available = False
        
        # 尝试初始化DeepSeek客户端
        if config.deepseek_api_key:
            try:
                self.deepseek_client = OpenAI(
                    api_key=config.deepseek_api_key,
                    base_url=config.deepseek_base_url,
                    timeout=config.timeout
                )
                self.deepseek_available = True
                logger.info("DeepSeek客户端初始化成功")
            except Exception as e:
                logger.warning(f"DeepSeek客户端初始化失败: {e}")
                self.deepseek_available = False
        
        # 检查千问API是否配置
        if config.qwen_api_key:
            self.qwen_available = True
            logger.info("千问API配置检查成功")
        else:
            logger.warning("千问API密钥未配置")
            self.qwen_available = False
            
        # 设置默认API
        if self.current_api == "auto":
            if self.deepseek_available:
                self.current_api = "deepseek"
            elif self.qwen_available:
                self.current_api = "qwen"
            else:
                raise Exception("没有可用的LLM API")
    
    def set_api(self, api_name: str) -> bool:
        """设置要使用的API，返回是否设置成功"""
        if api_name == "deepseek" and self.deepseek_available:
            self.current_api = "deepseek"
            return True
        elif api_name == "qwen" and self.qwen_available:
            self.current_api = "qwen"
            return True
        elif api_name == "blip2":
            self.current_api = "blip2"
            return True
        elif api_name == "auto":
            # 自动模式，优先使用DeepSeek
            if self.deepseek_available:
                self.current_api = "deepseek"
            elif self.qwen_available:
                self.current_api = "qwen"
            else:
                return False
            return True
        return False
    
    def get_current_api_info(self) -> Dict[str, Any]:
        """获取当前API的信息"""
        if self.current_api == "deepseek":
            return {
                "name": "DeepSeek",
                "model": self.config.deepseek_model,
                "available": self.deepseek_available
            }
        elif self.current_api == "qwen":
            return {
                "name": "千问",
                "model": self.config.qwen_model,
                "available": self.qwen_available
            }
        elif self.current_api == "blip2":
            return {
                "name": "本地BLIP2",
                "model": "Salesforce/blip2-opt-2.7b",
                "available": True
            }
        else:
            return {"name": "未知", "model": "未知", "available": False}
    
    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        """调用当前选择的LLM API"""
        if self.current_api == "deepseek" and self.deepseek_available:
            try:
                return self._call_deepseek_api(messages)
            except Exception as e:
                logger.warning(f"DeepSeek调用失败: {e}")
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "quota" in error_msg or "exceeded" in error_msg or "429" in error_msg:
                    self.deepseek_available = False
                    logger.warning("DeepSeek额度已用完，自动切换到千问")
                    if self.qwen_available:
                        self.current_api = "qwen"
                        return self._call_qwen_api(messages)
                raise e
        elif self.current_api == "qwen" and self.qwen_available:
            try:
                return self._call_qwen_api(messages)
            except Exception as e:
                logger.error(f"千问API调用失败: {e}")
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "quota" in error_msg or "exceeded" in error_msg or "429" in error_msg:
                    self.qwen_available = False
                    logger.warning("千问额度已用完，自动切换到DeepSeek")
                    if self.deepseek_available:
                        self.current_api = "deepseek"
                        return self._call_deepseek_api(messages)
                raise e
        raise Exception("当前没有可用的LLM API")
    
    def _call_deepseek_api(self, messages: List[Dict[str, str]]) -> str:
        for attempt in range(self.config.max_retries):
            try:
                response = self.deepseek_client.chat.completions.create(
                    model=self.config.deepseek_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"DeepSeek API调用异常 (尝试 {attempt+1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避策略
                    continue
                else:
                    raise e
        raise Exception("DeepSeek重试次数用尽")
    
    def _call_qwen_api(self, messages: List[Dict[str, str]]) -> str:
        headers = {
            "Authorization": f"Bearer {self.config.qwen_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.qwen_model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        for attempt in range(self.config.max_retries):
            try:
                resp = requests.post(
                    self.config.qwen_base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"千问API调用异常 (尝试 {attempt+1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避策略
                    continue
                else:
                    raise e
        raise Exception("千问API重试次数用尽")
    
    def _build_function_calling_prompt(self, user_query: str, functions: List[Dict]) -> str:
        function_desc = "可用函数:\n"
        for func in functions:
            function_desc += f"- {func['name']}: {func['description']}\n参数: {json.dumps(func['parameters'], ensure_ascii=False)}\n"
            
        prompt = f"""{function_desc}
        
        用户请求: {user_query}
        
        请分析用户需求，如果需要调用函数，请按以下格式回复：
        FUNCTION_CALL: {{"name": "函数名", "arguments": {{"参数名": "参数值"}}}}
        
        如果不需要调用函数，请直接回答用户问题。
        如果已经获得函数结果，请根据结果给出最终建议。
        请一步步思考并回答。"""
        return prompt

# 智能体调用核心类
class FunctionCallingAgent:
    def __init__(self, llm_adapter: LLMAdapter):
        self.llm = llm_adapter
    
    def _parse_function_call(self, response: str) -> Optional[Dict]:
        if "FUNCTION_CALL:" in response:
            try:
                func_call_str = response.split("FUNCTION_CALL:")[1].strip()
                func_call_str = func_call_str.split('\n')[0].strip()
                func_call = json.loads(func_call_str)
                return func_call
            except Exception as e:
                logger.warning(f"函数调用解析失败: {e}")
                logger.debug(f"尝试解析的字符串: {response}")
                return None
        return None
    
    def _execute_function(self, function_call: Dict) -> str:
        func_name = function_call.get("name")
        func_args = function_call.get("arguments", {})
        
        if func_name in AVAILABLE_FUNCTIONS:
            try:
                logger.info(f"执行函数: {func_name}({func_args})")
                result = AVAILABLE_FUNCTIONS[func_name](**func_args)
                logger.info(f"函数执行完成")
                return result
            except Exception as e:
                logger.error(f"函数执行失败: {e}")
                return json.dumps({"error": str(e)}, ensure_ascii=False)
        else:
            logger.warning(f"未知函数: {func_name}")
            return json.dumps({"error": f"未知函数: {func_name}"}, ensure_ascii=False)
    
    def process_query(self, user_query: str) -> str:
        """处理用户查询"""
        logger.info(f"用户查询: {user_query}")
        
        try:
            if self.llm.current_api == "blip2":
                query_lower = user_query.lower().strip()
                
                # 批量分析命令
                if query_lower.startswith("批量分析"):
                    folder_path = user_query[4:].strip()
                    if not folder_path:
                        return "错误: 请提供文件夹路径，例如：批量分析 /Users/xxx/Desktop/images/"
                    return analyze_multiple_images(folder_path)
                
                # 识别图片命令
                elif query_lower.startswith("识别图片"):
                    image_path = user_query[4:].strip()
                    if not image_path:
                        return "错误: 请提供图片路径，例如：识别图片 /Users/xxx/Desktop/test.jpg"
                    if not os.path.exists(image_path):
                        return f"错误: 图片路径 '{image_path}' 不存在"
                    return analyze_ev_image(image_path)
                
                # 直接输入路径的情况
                elif any(user_query.lower().endswith(fmt) for fmt in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']):
                    if not os.path.exists(user_query):
                        return f"错误: 图片路径 '{user_query}' 不存在"
                    return analyze_ev_image(user_query)
                
                # 未知命令
                else:
                    return "错误: 未知命令。支持的命令：\n- 识别图片 <路径>: 分析单张图片\n- 批量分析 <文件夹路径>: 批量分析文件夹中的图片\n- 直接输入图片路径: 快捷分析单张图片"
            
            # 其他模式，正常走LLM流程
            initial_prompt = self.llm._build_function_calling_prompt(user_query, FUNCTION_DESCRIPTIONS)
            messages = [{"role": "user", "content": initial_prompt}]
            
            response = self.llm.call_llm(messages)
            logger.info(f"模型初始响应: {response}")
            
            function_call = self._parse_function_call(response)
            
            if function_call:
                function_result = self._execute_function(function_call)
                
                follow_up_prompt = f"""之前的对话:
                用户请求: {user_query}
                你决定调用函数: {json.dumps(function_call, ensure_ascii=False)}
                函数返回结果: {function_result}
                
                现在请根据函数返回的结果，给出最终的建议回答。需要判断并省略无用的文字识别乱码，并对于提供的所有信息进行再识别和翻译（特别是与新能源车企方面有关）
                请用简洁的语言回答，直接给出有用的信息。"""
                
                messages = [{"role": "user", "content": follow_up_prompt}]
                final_response = self.llm.call_llm(messages)
                logger.info(f"模型最终响应: {final_response}")
                return final_response
            else:
                return response
                
        except Exception as e:
            logger.error(f"处理查询错误: {e}", exc_info=True)
            return f"抱歉，处理请求时出错: {e}"

#主程序入口
def show_menu(deepseek_available: bool, qwen_available: bool) -> str:
    """显示主菜单"""
    print("\n" + "=" * 50)
    print("函数调用智能体 - API选择 (CPU模式)")
    print("=" * 50)
    
    options = []
    if deepseek_available:
        options.append(("1", "使用 DeepSeek API"))
    if qwen_available:
        options.append(("2", "使用 千问 API"))
    if deepseek_available and qwen_available:
        options.append(("3", "自动模式 (优先DeepSeek，失败时切换 千问)"))
    options.append(("4", "本地blip2模式 (供测试使用)"))
    options.append(("exit", "退出程序"))
    
    for option, desc in options:
        print(f"{option}. {desc}")
    
    while True:
        choice = input("\n请选择模式: ").strip().lower()
        valid_choices = [option.lower() for option, _ in options]
        if choice in valid_choices:
            return choice
        print("无效输入，请重新选择")

def show_api_menu(deepseek_available: bool, qwen_available: bool) -> str:
    """显示API切换菜单"""
    print("\n" + "-" * 30)
    print("API切换菜单")
    print("-" * 30)
    
    options = []
    if deepseek_available:
        options.append(("1", "切换到 DeepSeek API"))
    if qwen_available:
        options.append(("2", "切换到 千问 API"))
    if deepseek_available and qwen_available:
        options.append(("3", "自动模式"))
    options.append(("4", "切换到 本地blip2模式"))
    options.append(("exit", "退出程序"))
    
    for option, desc in options:
        print(f"{option}. {desc}")
    
    while True:
        choice = input("\n请选择: ").strip().lower()
        valid_choices = [option.lower() for option, _ in options]
        if choice in valid_choices:
            return choice
        print("无效输入，请重新选择")

def main():
    global predictor
    
    print("=" * 60)
    print("本程序支持在DeepSeek和千问API之间灵活切换，构建一个基础的函数调用的智能体")
    print("CPU模式 - 图像处理速度较慢，请耐心等待")
    print("=" * 60)
    
    # 初始化情感分析模型
    model_path = 'D:\\OneDrive\\Desktop\\Epic2\\2.7图片加入\\sentiment_model_bert'
    try:
        predictor = SentimentPredictor(model_path)
    except Exception as e:
        print(f"情感分析模型初始化失败: {e}")
        sys.exit(1)
    
    config = load_config()
    
    try:
        llm_adapter = LLMAdapter(config)
        
        choice = show_menu(llm_adapter.deepseek_available, llm_adapter.qwen_available)
        
        if choice == "exit":
            print("再见！")
            sys.exit(0)
        
        # 根据用户选择设置API
        if choice == "1":
            llm_adapter.set_api("deepseek")
        elif choice == "2":
            llm_adapter.set_api("qwen")
        elif choice == "4":
            llm_adapter.set_api("blip2")
        else:  # 自动模式
            llm_adapter.set_api("auto")
        
        agent = FunctionCallingAgent(llm_adapter)
        
        while True:
            api_info = llm_adapter.get_current_api_info()
            print(f"\n当前使用: {api_info['name']} ({api_info['model']})")
            
            if api_info['name'] == "本地BLIP2":
                print("\n支持的命令:")
                print("- 识别图片 <路径>: 分析单张图片（含情感分析）")
                print("- 批量分析 <文件夹路径>: 批量分析文件夹中的图片（含情感分析）")
                print("- switch: 切换API")
                print("- exit: 退出程序")
                print("\n注意: CPU模式下处理速度较慢，批量处理限制为10张图片")
            
            else:
                print("\n支持的功能:")
                print("- 分析单张图片（含情感分析）")
                print("- 批量分析文件夹中的图片（含情感分析）")               
                print("- switch: 切换API")
                print("- exit: 退出程序")
                print("\n注意: CPU模式下处理速度较慢，批量处理限制为10张图片")
                
            query = input("\n请输入您的问题或命令: ").strip()
            
            if query.lower() == 'exit':
                print("再见！")
                break
            elif query.lower() == 'switch':
                api_choice = show_api_menu(llm_adapter.deepseek_available, llm_adapter.qwen_available)
                
                if api_choice == "exit":
                    print("再见！")
                    sys.exit(0)
                elif api_choice == "1":
                    success = llm_adapter.set_api("deepseek")
                    print("已切换到DeepSeek API" if success else "切换失败，DeepSeek API不可用")
                elif api_choice == "2":
                    success = llm_adapter.set_api("qwen")
                    print("已切换到千问 API" if success else "切换失败，千问 API不可用")
                elif api_choice == "4":
                    success = llm_adapter.set_api("blip2")
                    print("已切换到本地BLIP2模式" if success else "切换失败")
                else:  # 自动模式
                    llm_adapter.set_api("auto")
                    api_info = llm_adapter.get_current_api_info()
                    print(f"已切换到自动模式，当前使用: {api_info['name']}")
                
                continue
            
            if not query:
                print("请输入有效内容")
                continue
            
            print("正在处理，CPU模式下需要较长时间，请耐心等待...")
            
            try:
                answer = agent.process_query(query)
                print(f"\n识别结果: {answer}")
                
                # 显示后续操作选项
                print("\n操作选项:")
                print("1. 继续提问")
                print("2. 切换API")
                print("exit: 退出程序")
                
                while True:
                    op_choice = input("请选择操作 [1/2/exit]: ").strip().lower()
                    if op_choice == '1':
                        break
                    elif op_choice == '2':
                        api_choice = show_api_menu(llm_adapter.deepseek_available, llm_adapter.qwen_available)
                        
                        if api_choice == "exit":
                            print("再见！")
                            sys.exit(0)
                        elif api_choice == "1":
                            success = llm_adapter.set_api("deepseek")
                            print("已切换到DeepSeek API" if success else "切换失败，DeepSeek API不可用")
                        elif api_choice == "2":
                            success = llm_adapter.set_api("qwen")
                            print("已切换到千问 API" if success else "切换失败，千问 API不可用")
                        elif api_choice == "4":
                            success = llm_adapter.set_api("blip2")
                            print("已切换到本地BLIP2模式" if success else "切换失败")
                        else:  # 自动模式
                            llm_adapter.set_api("auto")
                            api_info = llm_adapter.get_current_api_info()
                            print(f"已切换到自动模式，当前使用: {api_info['name']}")
                        
                        break
                    elif op_choice == 'exit':
                        print("再见！")
                        sys.exit(0)
                    else:
                        print("无效输入，请重新选择")
                        
            except Exception as e:
                print(f"处理请求时出错: {e}")
                
    except Exception as e:
        print(f"程序初始化失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
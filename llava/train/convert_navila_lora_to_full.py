#!/usr/bin/env python3
"""
将 NaVILA LoRA checkpoint 转换为完整模型格式

用法:
python convert_navila_lora_to_full.py \
    --base_model NaVILA_new/ckpt/navila-llama3-8b-8f \
    --lora_checkpoint NaVILA_new/ckpt/navila-8b-8f-sft-epd-new/checkpoint-22400 \
    --output_dir NaVILA_new/ckpt/navila-8b-8f-sft-epd-new-converted

这个脚本会:
1. 加载基础NaVILA模型 (navila-llama3-8b-8f 格式)
2. 应用LoRA适配器权重 (从 navila-8b-8f-sft-epd-new 格式)
3. 合并所有权重并保存为完整模型格式

转换后的模型结构:
├── llm/                    # 语言模型权重
├── vision_tower/           # 视觉编码器权重  
├── mm_projector/          # 多模态投影器权重
└── config.json           # 主配置文件
"""

import argparse
import os
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoConfig
import shutil
import json
from safetensors.torch import save_file

# from llava.model import LlavaLlamaForCausalLM
from llava.model.builder import load_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(description="Convert NaVILA LoRA checkpoint to full model format")
    parser.add_argument("--base_model", type=str, required=True, 
                       help="Path to base NaVILA model (like navila-llama3-8b-8f)")
    parser.add_argument("--lora_checkpoint", type=str, required=True,
                       help="Path to LoRA checkpoint directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for the full model")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use for conversion")
    return parser.parse_args()


def convert_lora_to_full_model(base_model_path, lora_checkpoint_path, output_dir, device="cuda:0"):
    """
    将LoRA checkpoint转换为完整模型格式
    """
    print(f"开始转换: {lora_checkpoint_path} -> {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("1. 加载基础模型...")
    # 加载基础NaVILA模型 (使用现有的load_pretrained_model函数)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=base_model_path,
        model_name="navila",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    print(f"   ✅ 基础模型加载完成")
    print(f"   📋 模型类型: {type(model).__name__}")
    print(f"   📋 语言模型: {type(model.llm).__name__ if hasattr(model, 'llm') and model.llm else 'None'}")
    print(f"   📋 视觉塔: {type(model.vision_tower).__name__ if hasattr(model, 'vision_tower') and model.vision_tower else 'None'}")
    print(f"   📋 多模态投影器: {type(model.mm_projector).__name__ if hasattr(model, 'mm_projector') and model.mm_projector else 'None'}")
    print(f"   📋 分词器: {type(tokenizer).__name__}")
    
    print("2. 检查是否有non_lora_trainables.bin...")
    # 加载非LoRA可训练参数
    non_lora_path = os.path.join(lora_checkpoint_path, "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        print("   加载非LoRA可训练参数...")
        non_lora_trainables = torch.load(non_lora_path, map_location="cpu")
        
        # 处理键名格式
        if any(k.startswith("base_model.model.") for k in non_lora_trainables):
            non_lora_trainables = {
                (k[11:] if k.startswith("base_model.") else k): v 
                for k, v in non_lora_trainables.items()
            }
        
        if any(k.startswith("model.model.") for k in non_lora_trainables):
            non_lora_trainables = {
                (k[6:] if k.startswith("model.") else k): v 
                for k, v in non_lora_trainables.items()
            }
        
        # 加载非LoRA参数到模型
        model.load_state_dict(non_lora_trainables, strict=False)
        print("   非LoRA参数加载完成")
    
    print("3. 加载并合并LoRA权重...")
    # 检查是否有adapter配置
    adapter_config_path = os.path.join(lora_checkpoint_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        # 使用PEFT加载LoRA适配器
        model = PeftModel.from_pretrained(
            model,
            lora_checkpoint_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("   合并LoRA权重...")
        model = model.merge_and_unload()
        print("   LoRA权重合并完成")
    else:
        print("   未找到LoRA适配器配置，跳过LoRA合并")
    
    print("4. 保存完整模型...")
    # 保存模型的各个组件
    
    # 创建各个子目录
    llm_dir = os.path.join(output_dir, "llm")
    vision_tower_dir = os.path.join(output_dir, "vision_tower") 
    mm_projector_dir = os.path.join(output_dir, "mm_projector")
    
    os.makedirs(llm_dir, exist_ok=True)
    os.makedirs(vision_tower_dir, exist_ok=True)
    os.makedirs(mm_projector_dir, exist_ok=True)
    
    # 保存LLM部分
    print("   保存语言模型...")
    if hasattr(model, 'llm') and model.llm is not None:
        try:
            model.llm.save_pretrained(llm_dir)
            tokenizer.save_pretrained(llm_dir)
            print(f"     ✅ 保存语言模型: {type(model.llm).__name__}")
            print(f"     ✅ 保存分词器: {type(tokenizer).__name__}")
        except Exception as e:
            print(f"     ❌ 语言模型保存失败: {e}")
            raise
    else:
        print("   ⚠️  未找到语言模型组件")
    
    # 保存视觉塔
    print("   保存视觉编码器...")
    if hasattr(model, 'vision_tower') and model.vision_tower is not None:
        # SiglipVisionTower 的内部结构: vision_tower.vision_tower 是实际的模型
        if hasattr(model.vision_tower, 'vision_tower'):
            # 保存实际的视觉模型
            model.vision_tower.vision_tower.save_pretrained(vision_tower_dir)
            print(f"     ✅ 保存视觉模型: {type(model.vision_tower.vision_tower).__name__}")
        elif hasattr(model.vision_tower, 'save_pretrained'):
            # 如果是其他类型的视觉塔，直接保存
            model.vision_tower.save_pretrained(vision_tower_dir)
            print(f"     ✅ 保存视觉塔: {type(model.vision_tower).__name__}")
        else:
            print(f"     ⚠️  视觉塔类型 {type(model.vision_tower).__name__} 不支持 save_pretrained")
        
        # 保存图像处理器
        if hasattr(model.vision_tower, 'image_processor'):
            model.vision_tower.image_processor.save_pretrained(vision_tower_dir)
            print(f"     ✅ 保存图像处理器: {type(model.vision_tower.image_processor).__name__}")
    else:
        print("   ⚠️  未找到视觉编码器组件")
    
    # 保存多模态投影器
    print("   保存多模态投影器...")
    if hasattr(model, 'mm_projector') and model.mm_projector is not None:
        try:
            # 使用 PreTrainedModel 的标准保存方法
            model.mm_projector.save_pretrained(mm_projector_dir)
            print(f"     ✅ 保存多模态投影器: {type(model.mm_projector).__name__}")
        except Exception as e:
            print(f"     ⚠️  标准保存失败，尝试手动保存: {e}")
            # 备用方案：手动保存权重和配置
            save_file(model.mm_projector.state_dict(), os.path.join(mm_projector_dir, 'model.safetensors'))
            
            # 创建配置文件
            mm_projector_config = {
                "torch_dtype": "bfloat16",
                "architectures": ["MultimodalProjector"],
                "model_type": "v2l_projector"
            }
            with open(os.path.join(mm_projector_dir, "config.json"), "w") as f:
                json.dump(mm_projector_config, f, indent=2)
            print(f"     ✅ 手动保存完成")
    else:
        print("   ⚠️  未找到多模态投影器组件")
    
    # 复制主配置文件
    print("   复制配置文件...")
    if os.path.exists(os.path.join(base_model_path, "config.json")):
        shutil.copy2(
            os.path.join(base_model_path, "config.json"),
            os.path.join(output_dir, "config.json")
        )
    
    # 复制其他可能需要的文件
    for filename in [".gitattributes", "trainer_state.json"]:
        src_path = os.path.join(base_model_path, filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(output_dir, filename))
    
    print(f"✅ 转换完成! 完整模型已保存到: {output_dir}")
    print("\n模型结构:")
    print(f"  📁 {output_dir}/")
    print(f"    📁 llm/           # 语言模型权重")
    print(f"    📁 vision_tower/  # 视觉编码器权重") 
    print(f"    📁 mm_projector/  # 多模态投影器权重")
    print(f"    📄 config.json    # 主配置文件")


def main():
    args = parse_args()
    
    # 检查输入路径
    if not os.path.exists(args.base_model):
        raise ValueError(f"基础模型路径不存在: {args.base_model}")
    
    if not os.path.exists(args.lora_checkpoint):
        raise ValueError(f"LoRA checkpoint路径不存在: {args.lora_checkpoint}")
    
    # 执行转换
    convert_lora_to_full_model(
        base_model_path=args.base_model,
        lora_checkpoint_path=args.lora_checkpoint,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main() 
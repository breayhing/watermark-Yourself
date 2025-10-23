#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中心区域水印工具
在图片中心很小的区域嵌入隐形水印，最大程度减少对图片的影响
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageCms
import io
import warnings
from blind_watermark import WaterMark


class CenterWatermark:
    """中心区域水印工具"""
    
    def __init__(self):
        warnings.filterwarnings("ignore", message=".*libpng warning.*")
    
    def fix_color_profile(self, input_path, output_path=None):
        """修复颜色配置文件"""
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_fixed{ext}"
        
        try:
            with Image.open(input_path) as img:
                if 'icc_profile' in img.info:
                    try:
                        srgb_profile = ImageCms.createProfile('sRGB')
                        current_profile = ImageCms.ImageCmsProfile(io.BytesIO(img.info['icc_profile']))
                        transform = ImageCms.buildTransformFromOpenProfiles(
                            current_profile, srgb_profile, 'RGB', 'RGB'
                        )
                        
                        if img.mode == 'RGBA':
                            alpha = img.split()[-1]
                            img_rgb = img.convert('RGB')
                            img_converted = ImageCms.applyTransform(img_rgb, transform)
                            img_converted = img_converted.convert('RGBA')
                            img_converted.putalpha(alpha)
                        else:
                            img_converted = ImageCms.applyTransform(img, transform)
                        
                        img_converted.save(output_path, quality=100, optimize=False)
                        
                    except Exception as e:
                        img.save(output_path, quality=100, optimize=False)
                else:
                    img.save(output_path, quality=100, optimize=False)
                
                return output_path
                
        except Exception as e:
            print(f"处理失败: {e}")
            return input_path
    
    def calculate_required_size(self, watermark_text):
        """计算水印所需的最小尺寸"""
        # 估算水印大小（字节）
        watermark_bytes = len(watermark_text.encode('utf-8'))
        print(f"   📝 水印字节数: {watermark_bytes}")
        
        # 根据实际失败案例重新校准：
        # 64x64区域实际只能容纳约7-8字节，不是65字节！
        # 所以每个像素实际存储密度约为 8/4096 ≈ 0.002字节
        # 为了保证成功，使用更保守的估计
        
        bytes_per_pixel = 0.002  # 非常保守的估计
        required_pixels = watermark_bytes / bytes_per_pixel
        
        # 计算边长，向上取整
        required_side = int(np.ceil(np.sqrt(required_pixels)))
        
        print(f"   🧮 估算需要像素数: {int(required_pixels)}")
        print(f"   📐 初步计算边长: {required_side}")
        
        # 确保是16的倍数（便于处理）并设置合理的范围
        required_side = ((required_side + 15) // 16) * 16
        required_side = max(64, min(512, required_side))
        
        print(f"   🎯 调整后推荐边长: {required_side}")
        
        return required_side
    
    def add_center_watermark(self, image_path, watermark_text, output_path, center_ratio=0.01, auto_adjust=True):
        """
        在图片中心区域添加隐形水印
        
        Args:
            image_path: 输入图片路径
            watermark_text: 水印文字
            output_path: 输出路径
            center_ratio: 中心区域大小比例 (0.005-0.1，默认0.01即1%)
            auto_adjust: 是否自动调整区域大小以适应水印容量
        """
        try:
            print(f"\n🎯 在中心区域添加隐形水印 (目标比例: {center_ratio*100:.2f}%)...")
            
            # 修复颜色配置文件
            base, ext = os.path.splitext(image_path)
            temp_file = f"{base}_temp_fixed{ext}"
            processed_image = self.fix_color_profile(image_path, temp_file)
            
            # 读取图片
            img = cv2.imread(processed_image)
            if img is None:
                raise ValueError(f"无法读取图片: {processed_image}")
            
            height, width = img.shape[:2]
            print(f"   📐 图片尺寸: {width} x {height}")
            
            # 计算所需最小尺寸
            required_size = self.calculate_required_size(watermark_text)
            print(f"   🎯 推荐最小尺寸: {required_size}x{required_size} 像素")
            
            # 计算中心区域大小
            if auto_adjust:
                # 自动调整：优先使用计算出的最小需求，但不能超过图片的20%
                max_allowed = min(int(width * 0.2), int(height * 0.2))
                center_size = min(required_size, max_allowed)
                center_h = center_w = center_size
                
                print(f"   📊 自动调整模式: 使用尺寸 {center_size}x{center_size}")
                
                # 如果计算的尺寸太大，警告用户
                if required_size > max_allowed:
                    print(f"   ⚠️  水印内容较多，推荐尺寸({required_size})超过限制，使用最大允许尺寸({max_allowed})")
                    print(f"   💡 建议：缩短水印文本或使用更大的图片")
            else:
                # 用户固定设置
                center_h = max(64, int(height * center_ratio))
                center_w = max(64, int(width * center_ratio))
                
                print(f"   📊 固定比例模式: 使用尺寸 {center_w}x{center_h}")
                
                # 检查用户设置是否足够
                if max(center_h, center_w) < required_size:
                    print(f"   ⚠️  用户设置的区域({max(center_h, center_w)})可能不足以容纳水印({required_size})")
            
            # 确保区域是偶数（有些算法要求）
            center_h = center_h + (center_h % 2)
            center_w = center_w + (center_w % 2)
            
            # 限制最大区域不超过图片的20%（增加上限以容纳更多水印）
            max_h = int(height * 0.2)
            max_w = int(width * 0.2)
            center_h = min(center_h, max_h)
            center_w = min(center_w, max_w)
            
            # 计算实际影响比例
            actual_ratio = (center_w * center_h) / (width * height)
            
            # 计算中心区域的起始位置
            start_y = (height - center_h) // 2
            start_x = (width - center_w) // 2
            end_y = start_y + center_h
            end_x = start_x + center_w
            
            print(f"   📍 中心区域: ({start_x}, {start_y}) 到 ({end_x}, {end_y})")
            print(f"   📏 区域大小: {center_w} x {center_h} 像素")
            print(f"   📊 实际影响比例: {actual_ratio * 100:.3f}%")
            
            if auto_adjust and actual_ratio > center_ratio * 3:
                print(f"   ⚠️  为容纳水印，区域已自动扩大到 {actual_ratio * 100:.2f}%")
            
            # 预估容量并给出提醒（使用保守的公式）
            estimated_capacity = (center_w * center_h) * 0.002  # 使用非常保守的估算
            watermark_size = len(watermark_text.encode('utf-8'))
            capacity_ratio = watermark_size / estimated_capacity
            
            print(f"   📊 预估容量: {estimated_capacity:.1f} 字节")
            print(f"   📝 水印大小: {watermark_size} 字节")
            print(f"   📈 容量使用率: {capacity_ratio * 100:.1f}%")
            
            if capacity_ratio > 0.8:
                print(f"   ⚠️  容量使用率较高，可能影响水印质量")
            elif capacity_ratio > 1.0:
                print(f"   🚫 容量不足，水印可能嵌入失败")
            
            # 提取中心区域
            center_region = img[start_y:end_y, start_x:end_x].copy()
            
            # 创建临时文件用于中心区域水印
            center_temp = "temp_center.png"
            cv2.imwrite(center_temp, center_region)
            
            # 对中心区域添加隐形水印
            bwm = WaterMark(password_img=1, password_wm=1)
            bwm.read_img(center_temp)
            bwm.read_wm(watermark_text, mode='str')
            
            center_watermarked = "temp_center_watermarked.png"
            bwm.embed(center_watermarked)
            
            # 读取水印后的中心区域
            watermarked_center = cv2.imread(center_watermarked)
            
            if watermarked_center is None:
                raise ValueError("水印嵌入失败")
            
            # 将水印后的中心区域放回原图
            result_img = img.copy()
            result_img[start_y:end_y, start_x:end_x] = watermarked_center
            
            # 保存结果
            cv2.imwrite(output_path, result_img)
            
            # 清理临时文件
            for temp in [temp_file, center_temp, center_watermarked]:
                if os.path.exists(temp):
                    os.remove(temp)
            
            print(f"   ✅ 中心水印添加成功: {output_path}")
            print(f"   💡 水印位置: 图片正中心")
            print(f"   🔢 水印容量: {len(bwm.wm_bit)} 位")
            
            return True, f"中心水印添加成功！实际影响区域: {actual_ratio*100:.3f}%", len(bwm.wm_bit)
            
        except Exception as e:
            # 清理临时文件
            for temp in ['temp_file', 'center_temp', 'center_watermarked']:
                if temp in locals() and os.path.exists(locals()[temp]):
                    os.remove(locals()[temp])
            return False, f"添加中心水印失败: {str(e)}", None
    
    def extract_center_watermark(self, watermarked_image_path, wm_shape, center_ratio=0.01, watermark_text=""):
        """
        从中心区域提取隐形水印
        
        Args:
            watermarked_image_path: 含水印的图片路径
            wm_shape: 水印长度（之前嵌入时返回的值）
            center_ratio: 中心区域大小比例（必须与嵌入时相同）
            watermark_text: 原始水印文本（用于计算区域大小，必须与嵌入时相同）
        """
        try:
            print(f"\n🔍 从中心区域提取隐形水印...")
            
            # 修复颜色配置文件
            base, ext = os.path.splitext(watermarked_image_path)
            temp_file = f"{base}_temp_fixed{ext}"
            processed_image = self.fix_color_profile(watermarked_image_path, temp_file)
            
            # 读取图片
            img = cv2.imread(processed_image)
            if img is None:
                raise ValueError(f"无法读取图片: {processed_image}")
            
            height, width = img.shape[:2]
            
            # 使用与添加水印时完全相同的区域计算逻辑
            if watermark_text:
                required_size = self.calculate_required_size(watermark_text)
                center_h = max(required_size, int(height * center_ratio))
                center_w = max(required_size, int(width * center_ratio))
            else:
                # 如果没有提供原始文本，使用默认方法
                center_h = max(64, int(height * center_ratio))
                center_w = max(64, int(width * center_ratio))
            
            # 确保区域是偶数（与添加时保持一致）
            center_h = center_h + (center_h % 2)
            center_w = center_w + (center_w % 2)
            
            # 限制最大区域不超过图片的20%（与添加时保持一致）
            max_h = int(height * 0.2)
            max_w = int(width * 0.2)
            center_h = min(center_h, max_h)
            center_w = min(center_w, max_w)
            
            start_y = (height - center_h) // 2
            start_x = (width - center_w) // 2
            end_y = start_y + center_h
            end_x = start_x + center_w
            
            print(f"   📏 提取区域: {center_w} x {center_h} 像素")
            print(f"   📍 提取坐标: ({start_x}, {start_y}) 到 ({end_x}, {end_y})")
            
            # 提取中心区域
            center_region = img[start_y:end_y, start_x:end_x].copy()
            
            # 保存中心区域到临时文件
            center_temp = "temp_center_extract.png"
            cv2.imwrite(center_temp, center_region)
            
            # 提取水印
            bwm = WaterMark(password_img=1, password_wm=1)
            extracted_watermark = bwm.extract(center_temp, wm_shape=wm_shape, mode='str')
            
            # 清理临时文件
            for temp in [temp_file, center_temp]:
                if os.path.exists(temp):
                    os.remove(temp)
            
            print(f"   ✅ 水印提取成功: {extracted_watermark}")
            return True, extracted_watermark
            
        except Exception as e:
            # 清理临时文件
            for temp in ['temp_file', 'center_temp']:
                if temp in locals() and os.path.exists(locals()[temp]):
                    os.remove(locals()[temp])
            return False, f"提取中心水印失败: {str(e)}"
    
    def demo_center_watermark(self, input_image, watermark_text="© 2025"):
        """演示中心水印的完整流程"""
        print("=" * 60)
        print("🎯 中心区域水印演示")
        print("=" * 60)
        
        if not os.path.exists(input_image):
            print(f"❌ 找不到输入文件: {input_image}")
            return
        
        # 测试用例：野草宣传部
        test_cases = [
            {"text": "野草宣传部", "ratio": 0.01, "description": "野草宣传部水印"},
        ]
        
        for i, case in enumerate(test_cases):
            print(f"\n📊 测试案例 {i+1}: {case['description']}")
            print(f"   水印内容: '{case['text']}'")
            output_path = f"center_watermark_case{i+1}.png"
            
            # 添加水印（启用自动调整）
            success, message, wm_shape = self.add_center_watermark(
                input_image, case['text'], output_path, case['ratio'], auto_adjust=True
            )
            
            if success:
                print(f"   ✅ {message}")
                
                # 提取水印验证
                extract_success, extracted_text = self.extract_center_watermark(
                    output_path, wm_shape, case['ratio'], case['text']
                )
                
                if extract_success:
                    print(f"   🔍 提取验证: '{extracted_text}'")
                    if extracted_text.strip() == case['text'].strip():
                        print(f"   ✅ 水印完整性验证通过")
                    else:
                        print(f"   ⚠️  水印可能有损失，但仍可识别")
                else:
                    print(f"   ❌ 提取失败")
            else:
                print(f"   ❌ {message}")
        
        print("\n" + "=" * 60)


def main():
    """主函数"""
    watermark_tool = CenterWatermark()
    
    # 设置输入图片
    input_image = "frost.png"
    
    # 运行演示
    watermark_tool.demo_center_watermark(input_image, "野草宣传部")


if __name__ == "__main__":
    main()
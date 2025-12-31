#!/usr/bin/env python3
"""
自动解决Git合并冲突的脚本
接受传入的更改（即 a6754c6 提交的内容）
"""

import os
import re
import glob

def resolve_conflicts_in_file(file_path):
    """解决单个文件中的合并冲突"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否有冲突标记
        if '<<<<<<< HEAD' not in content:
            return False
        
        print(f"正在解决冲突: {file_path}")
        
        # 使用正则表达式匹配冲突块
        # 模式: <<<<<<< HEAD ... ======= ... >>>>>>> commit_hash
        conflict_pattern = r'<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> [^\n]+\n?'
        
        # 替换为传入的更改（第二个捕获组）
        resolved_content = re.sub(conflict_pattern, r'\2', content, flags=re.DOTALL)
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(resolved_content)
        
        print(f"✓ 已解决冲突: {file_path}")
        return True
        
    except Exception as e:
        print(f"✗ 解决冲突失败 {file_path}: {e}")
        return False

def find_conflicted_files():
    """查找所有包含冲突标记的文件"""
    conflicted_files = []
    
    # 搜索所有文件
    for root, dirs, files in os.walk('.'):
        # 跳过一些目录
        dirs[:] = [d for d in dirs if not d.startswith('.git') and d != 'node_modules' and d != '__pycache__']
        
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if '<<<<<<< HEAD' in content:
                        conflicted_files.append(file_path)
            except:
                continue
    
    return conflicted_files

def main():
    print("开始解决Git合并冲突...")
    
    # 查找冲突文件
    conflicted_files = find_conflicted_files()
    
    if not conflicted_files:
        print("没有发现合并冲突文件")
        return
    
    print(f"发现 {len(conflicted_files)} 个冲突文件:")
    for file in conflicted_files:
        print(f"  - {file}")
    
    print("\n开始解决冲突...")
    
    resolved_count = 0
    for file_path in conflicted_files:
        if resolve_conflicts_in_file(file_path):
            resolved_count += 1
    
    print(f"\n完成! 成功解决了 {resolved_count}/{len(conflicted_files)} 个文件的冲突")
    
    if resolved_count == len(conflicted_files):
        print("\n所有冲突已解决。您现在可以运行以下命令完成合并:")
        print("git add .")
        print("git commit -m 'resolve merge conflicts'")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
批量将 HEROUI 组件替换为 MUI 组件
"""

import re
import os
from pathlib import Path

# 组件映射
COMPONENT_MAPPING = {
    # 导入语句
    "from '@heroui/react'": "from '@mui/material'",
    
    # 组件替换
    "Card": "Card",
    "CardHeader": "CardHeader",  # MUI 使用 CardHeader 或 Typography
    "CardBody": "CardContent",
    "CardFooter": "CardActions",
    
    "Button": "Button",
    "onPress": "onClick",
    
    "Input": "TextField",
    "Textarea": "TextField",  # 需要 multiline prop
    "onValueChange": "onChange",
    
    "Select": "Select",
    "SelectItem": "MenuItem",
    
    "Switch": "Switch",
    
    "Slider": "Slider",
    
    "Chip": "Chip",
    
    "Progress": "LinearProgress",
    
    "Table": "Table",
    "TableHeader": "TableHead",
    "TableColumn": "TableCell",
    "TableBody": "TableBody",
    "TableRow": "TableRow",
    "TableCell": "TableCell",
    
    "Modal": "Dialog",
    "ModalContent": "DialogContent",
    "ModalHeader": "DialogTitle",
    "ModalBody": "DialogContent",
    "ModalFooter": "DialogActions",
    "useDisclosure": "",  # 需要手动处理
    
    "Tabs": "Tabs",
    "Tab": "Tab",
    
    "Divider": "Divider",
    
    "Avatar": "Avatar",
    
    "Tooltip": "Tooltip",
    
    "Spacer": "Box",  # 需要 sx={{ flexGrow: 1 }}
}

def replace_components_in_file(file_path):
    """替换文件中的组件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 替换导入语句
    if "@heroui/react" in content:
        content = content.replace(
            "from '@heroui/react'",
            "from '@mui/material'"
        )
        # 添加必要的 MUI 导入
        if "Card" in content and "Card" not in content[:500]:
            content = content.replace(
                "from '@mui/material'",
                "from '@mui/material'\nimport { Card, CardContent, CardActions, CardHeader } from '@mui/material'"
            )
    
    # 替换组件使用
    # CardBody -> CardContent
    content = re.sub(r'<CardBody\s', '<CardContent ', content)
    content = re.sub(r'</CardBody>', '</CardContent>', content)
    
    # CardFooter -> CardActions
    content = re.sub(r'<CardFooter\s', '<CardActions ', content)
    content = re.sub(r'</CardFooter>', '</CardActions>', content)
    
    # onPress -> onClick
    content = re.sub(r'\bonPress\s*=', 'onClick=', content)
    
    # SelectItem -> MenuItem
    content = re.sub(r'<SelectItem\s', '<MenuItem ', content)
    content = re.sub(r'</SelectItem>', '</MenuItem>', content)
    
    # Progress -> LinearProgress
    content = re.sub(r'<Progress\s', '<LinearProgress ', content)
    content = re.sub(r'</Progress>', '</LinearProgress>', content)
    
    # TableHeader -> TableHead
    content = re.sub(r'<TableHeader\s', '<TableHead ', content)
    content = re.sub(r'</TableHeader>', '</TableHead>', content)
    
    # TableColumn -> TableCell (在 TableHead 中)
    # 这个需要更仔细的处理
    
    # Modal -> Dialog
    content = re.sub(r'<Modal\s', '<Dialog ', content)
    content = re.sub(r'</Modal>', '</Dialog>', content)
    content = re.sub(r'<ModalContent\s', '<DialogContent ', content)
    content = re.sub(r'</ModalContent>', '</DialogContent>', content)
    content = re.sub(r'<ModalHeader\s', '<DialogTitle ', content)
    content = re.sub(r'</ModalHeader>', '</DialogTitle>', content)
    content = re.sub(r'<ModalBody\s', '<DialogContent ', content)
    content = re.sub(r'</ModalBody>', '</DialogContent>', content)
    content = re.sub(r'<ModalFooter\s', '<DialogActions ', content)
    content = re.sub(r'</ModalFooter>', '</DialogActions>', content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"已更新: {file_path}")
        return True
    return False

def main():
    """主函数"""
    src_dir = Path("src")
    
    # 查找所有 TypeScript/TSX 文件
    tsx_files = list(src_dir.rglob("*.tsx")) + list(src_dir.rglob("*.ts"))
    
    updated_count = 0
    
    for file_path in tsx_files:
        if "@heroui" in file_path.read_text(encoding='utf-8'):
            if replace_components_in_file(file_path):
                updated_count += 1
    
    print(f"\n总共更新了 {updated_count} 个文件")

if __name__ == "__main__":
    main()

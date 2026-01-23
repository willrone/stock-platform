# 回测平台PPT演示文档目录

本目录包含回测平台的完整PPT演示文档和相关资源。

---

## 📁 文件说明

### 1. **回测平台PPT_Marp.md** ⭐ 主要PPT文件
- **用途**: Marp格式的完整PPT演示文稿
- **页数**: 约40页
- **内容**: 
  - 平台概述
  - 回测流程详解（6大阶段）
  - 策略详解（10+种策略）
  - 使用方法（创建任务、监控进度、查看结果）
  - 实战案例
  - 最佳实践
- **特点**: 可直接生成PPT演示文稿
- **使用方法**: 
  - 在线: https://marp.app/ → 打开文件 → 导出PDF/PPTX
  - VS Code: 安装"Marp for VS Code"插件 → 预览 → 导出
  - 命令行: `marp 回测平台PPT_Marp.md --pdf` 或 `--pptx`

---

### 2. **回测平台PPT_简洁版.md**
- **用途**: 精简版PPT，适合快速演示
- **页数**: 22页
- **内容**: 核心内容精简版，突出关键信息
- **适用场景**: 
  - 时间有限的演示
  - 快速培训
  - 概览性介绍

---

### 3. **回测平台流程图_Mermaid.md**
- **用途**: 所有流程图的Mermaid源码
- **内容**: 包含8个详细的流程图
  1. **回测流程整体图** - 展示完整的6阶段回测流程
  2. **回测执行详细流程图** - 阶段4的详细执行步骤
  3. **策略分类架构图** - 所有策略的分类和关系
  4. **移动平均策略流程图** - 移动平均策略的信号生成逻辑
  5. **RSI策略流程图** - RSI策略的复杂信号生成逻辑
  6. **组合策略信号整合流程图** - 多策略信号整合方法
  7. **数据加载流程图** - 数据加载的并行处理流程
  8. **任务创建流程图** - 从用户操作到任务创建的完整流程
- **使用方法**:
  - 在线查看: https://mermaid.live/ → 复制代码 → 查看/导出
  - VS Code: 安装"Markdown Preview Mermaid Support"插件
  - 导出图片: 使用mermaid-cli或在线工具导出PNG/SVG

---

### 4. **回测平台PPT讲解.md**
- **用途**: 详细的讲解文档（原始完整版）
- **内容**: 包含所有细节的完整文档
- **特点**: 
  - 最详细的说明文档
  - 包含代码示例和配置说明
  - 适合作为参考文档
- **适用场景**:
  - 详细学习平台功能
  - 查找具体配置方法
  - 作为技术文档参考

---

## 🚀 快速开始

### 生成PPT演示文稿

#### 方法1: 在线使用Marp（最简单）⭐
1. 访问 https://marp.app/
2. 打开 `回测平台PPT_Marp.md` 文件
3. 点击右上角 "Export" 按钮
4. 选择导出格式: PDF 或 PPTX

#### 方法2: 使用VS Code
1. 安装 "Marp for VS Code" 插件
2. 打开 `回测平台PPT_Marp.md`
3. 点击右上角的预览按钮（或按 `Ctrl+Shift+V`）
4. 使用命令面板 (`Ctrl+Shift+P`) 输入 "Marp: Export slide deck"
5. 选择导出格式: PDF/PPTX/HTML

#### 方法3: 命令行工具
```bash
# 安装Marp CLI
npm install -g @marp-team/marp-cli

# 生成PDF
marp docs/presentation/回测平台PPT_Marp.md --pdf -o 回测平台PPT.pdf

# 生成PPTX (需要pandoc)
marp docs/presentation/回测平台PPT_Marp.md --pptx -o 回测平台PPT.pptx

# 生成HTML
marp docs/presentation/回测平台PPT_Marp.md --html -o 回测平台PPT.html
```

---

### 查看和导出流程图

#### 在线查看（推荐）
1. 访问 https://mermaid.live/
2. 打开 `回测平台流程图_Mermaid.md`
3. 复制任意一个流程图代码
4. 粘贴到在线编辑器中
5. 点击 "Actions" → "Download PNG" 或 "Download SVG"

#### VS Code查看
1. 安装 "Markdown Preview Mermaid Support" 插件
2. 打开 `回测平台流程图_Mermaid.md`
3. 使用Markdown预览功能查看
4. 右键点击图表可以导出为图片

#### 命令行导出
```bash
# 安装mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# 导出所有流程图
mmdc -i docs/presentation/回测平台流程图_Mermaid.md -o docs/presentation/流程图.png
```

---

## 📊 文档结构

```
docs/presentation/
├── README.md                          # 本说明文件
├── 回测平台PPT_Marp.md                # 完整PPT (Marp格式) ⭐
├── 回测平台PPT_简洁版.md              # 精简版PPT
├── 回测平台流程图_Mermaid.md          # 流程图源码
└── 回测平台PPT讲解.md                 # 详细讲解文档
```

---

## 🎯 使用建议

### 演示准备
1. **主要演示**: 使用 `回测平台PPT_Marp.md` 生成PPT
2. **快速演示**: 使用 `回测平台PPT_简洁版.md`
3. **流程图**: 从 `回测平台流程图_Mermaid.md` 导出需要的流程图
4. **参考文档**: 使用 `回测平台PPT讲解.md` 作为详细参考

### 演示流程
1. **准备阶段**:
   - 生成PPT文件（PDF或PPTX）
   - 导出需要的流程图为图片
   - 根据实际情况调整内容

2. **演示阶段**:
   - 使用生成的PPT进行演示
   - 在需要时展示流程图
   - 参考详细文档回答提问

3. **后续支持**:
   - 将详细文档分享给需要深入了解的听众
   - 流程图可用于技术文档

---

## 🎨 自定义和修改

### 修改PPT内容
- 直接编辑 `回测平台PPT_Marp.md` 文件
- Marp使用Markdown语法，易于编辑
- 使用 `---` 分隔幻灯片页面

### 修改流程图
- 编辑 `回测平台流程图_Mermaid.md` 中的Mermaid代码
- 在线工具 https://mermaid.live/ 可以实时预览
- 修改后重新导出图片

### 自定义样式
在Marp文件顶部修改样式：
```markdown
---
marp: true
theme: default  # 可选: default, gaia, uncover
paginate: true
header: '您的标题'
footer: '您的页脚'
style: |
  section {
    font-family: 'Microsoft YaHei', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
  }
---
```

---

## 📚 相关资源

- **Marp官网**: https://marp.app/
- **Mermaid官网**: https://mermaid.js.org/
- **Marp在线编辑器**: https://marp.app/
- **Mermaid在线编辑器**: https://mermaid.live/

---

## ✅ 检查清单

在演示前，请确认：

- [ ] 已生成PPT文件（PDF或PPTX）
- [ ] 流程图已导出为图片（如需要）
- [ ] 内容已根据实际情况调整
- [ ] 示例数据已更新（股票代码、日期等）
- [ ] 已测试所有链接和代码示例
- [ ] 已准备演示用的回测案例

---

## 📝 更新日志

- **2026-01-23**: 创建初始版本
  - 完整PPT文档（Marp格式）
  - 精简版PPT
  - 8个详细流程图
  - 详细讲解文档

---

**如有问题或需要修改，请联系开发团队。**

**祝您演示顺利！** 🚀

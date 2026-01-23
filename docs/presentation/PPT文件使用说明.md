# PPT文件使用说明

## 📁 文件清单

本次为您创建了以下文件：

### 1. **回测平台PPT_Marp.md** ⭐ 推荐
- **格式**: Marp格式（Markdown演示文稿）
- **页数**: 约40页
- **内容**: 完整的PPT内容，包含所有策略详解和使用方法
- **特点**: 可直接生成PPT演示文稿

### 2. **回测平台PPT_简洁版.md**
- **格式**: Markdown格式
- **页数**: 22页
- **内容**: 精简版PPT，突出核心内容
- **特点**: 适合快速演示

### 3. **回测平台流程图_Mermaid.md**
- **格式**: Mermaid流程图源码
- **内容**: 8个详细的流程图
  - 回测流程整体图
  - 回测执行详细流程图
  - 策略分类架构图
  - 移动平均策略流程图
  - RSI策略流程图
  - 组合策略信号整合流程图
  - 数据加载流程图
  - 任务创建流程图

### 4. **回测平台PPT讲解.md**
- **格式**: 详细文档格式
- **内容**: 完整的讲解文档，包含所有细节

---

## 🚀 快速开始

### 方法1: 使用Marp生成PPT（推荐）

#### 步骤1: 安装Marp CLI
```bash
npm install -g @marp-team/marp-cli
```

#### 步骤2: 生成PPT
```bash
# 生成PDF
marp 回测平台PPT_Marp.md --pdf

# 生成HTML
marp 回测平台PPT_Marp.md --html

# 生成PowerPoint (需要安装pandoc)
marp 回测平台PPT_Marp.md --pptx
```

#### 步骤3: 在线使用Marp
1. 访问 https://marp.app/
2. 打开 `回测平台PPT_Marp.md` 文件
3. 点击"Export"导出为PDF/PPTX

### 方法2: 使用VS Code + Marp插件

#### 步骤1: 安装插件
1. 打开VS Code
2. 搜索并安装 "Marp for VS Code" 插件

#### 步骤2: 预览和导出
1. 打开 `回测平台PPT_Marp.md`
2. 点击右上角的预览按钮
3. 使用命令面板 (Ctrl+Shift+P) 输入 "Marp: Export slide deck"
4. 选择导出格式 (PDF/PPTX/HTML)

### 方法3: 查看流程图

#### 在线查看（最简单）
1. 访问 https://mermaid.live/
2. 复制 `回测平台流程图_Mermaid.md` 中的任意一个流程图代码
3. 粘贴到编辑器中即可查看和导出

#### VS Code查看
1. 安装 "Markdown Preview Mermaid Support" 插件
2. 打开 `回测平台流程图_Mermaid.md`
3. 使用Markdown预览功能查看

#### 导出为图片
```bash
# 安装mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# 导出单个流程图
mmdc -i 回测平台流程图_Mermaid.md -o 流程图.png

# 或者使用在线工具
# https://mermaid.live/ → 复制代码 → 导出PNG/SVG
```

---

## 📊 流程图使用指南

### 流程图列表

1. **回测流程整体图** - 展示完整的6阶段回测流程
2. **回测执行详细流程图** - 阶段4的详细执行步骤
3. **策略分类架构图** - 所有策略的分类和关系
4. **移动平均策略流程图** - 移动平均策略的信号生成逻辑
5. **RSI策略流程图** - RSI策略的复杂信号生成逻辑
6. **组合策略信号整合流程图** - 多策略信号整合方法
7. **数据加载流程图** - 数据加载的并行处理流程
8. **任务创建流程图** - 从用户操作到任务创建的完整流程

### 如何将流程图插入PPT

1. **方法1: 导出为图片**
   - 使用 mermaid.live 或 mermaid-cli 导出PNG
   - 在PPT中插入图片

2. **方法2: 使用Marp**
   - Marp支持直接渲染Mermaid代码
   - 在Marp文件中直接使用流程图代码即可

3. **方法3: 截图**
   - 在支持Mermaid的工具中查看
   - 截图插入PPT

---

## 🎨 自定义样式

### 修改Marp主题

编辑 `回测平台PPT_Marp.md` 文件顶部的样式：

```markdown
---
marp: true
theme: default  # 可改为: default, gaia, uncover
paginate: true
header: '量化交易回测平台使用指南'
footer: '© 2026 回测平台'
style: |
  section {
    font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
  }
---
```

### 可用主题
- `default` - 默认主题
- `gaia` - 简洁主题
- `uncover` - 现代主题

---

## 📝 内容修改建议

### 根据实际需求调整

1. **简化内容**: 如果演示时间有限，使用 `回测平台PPT_简洁版.md`
2. **添加案例**: 在实战案例部分添加您的实际回测案例
3. **调整策略**: 根据平台实际支持的策略调整策略列表
4. **更新数据**: 更新示例中的股票代码、日期等数据

### 自定义页面

在Marp文件中，使用以下语法创建新页面：

```markdown
---

## 新页面标题

内容...

---
```

---

## 🔧 常见问题

### Q: Marp生成的PPT格式不对？
A: 确保安装了最新版本的Marp CLI，或使用在线版本 https://marp.app/

### Q: 流程图显示不出来？
A: 
- 确保使用支持Mermaid的工具
- 检查代码格式是否正确
- 尝试使用在线工具 https://mermaid.live/

### Q: 如何修改流程图颜色？
A: 在Mermaid代码中使用 `style` 指令修改，例如：
```mermaid
style Start fill:#667eea,stroke:#333,stroke-width:2px,color:#fff
```

### Q: 如何将Marp PPT转换为PowerPoint？
A: 
- 方法1: 使用Marp CLI的 `--pptx` 选项（需要pandoc）
- 方法2: 先导出为PDF，再用PowerPoint打开并转换
- 方法3: 使用在线Marp工具导出

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
- [ ] 示例数据已更新
- [ ] 已测试所有链接和代码示例
- [ ] 已准备演示用的回测案例

---

## 🎯 推荐使用流程

1. **准备阶段**:
   - 使用 `回测平台PPT_Marp.md` 生成PPT
   - 从 `回测平台流程图_Mermaid.md` 导出需要的流程图
   - 根据实际情况调整内容

2. **演示阶段**:
   - 使用生成的PPT进行演示
   - 在需要时展示流程图
   - 参考 `回测平台PPT讲解.md` 获取详细说明

3. **后续支持**:
   - 将 `回测平台PPT讲解.md` 作为参考文档
   - 流程图可用于技术文档

---

**祝您演示顺利！** 🚀

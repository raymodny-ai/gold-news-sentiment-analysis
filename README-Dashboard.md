# 📊 仪表板使用指南

## 黄金价格预测模型的新闻情绪分析系统 - 仪表板

### 🎯 仪表板功能

仪表板提供以下功能：

- **📈 实时概览**: 系统状态、新闻数量、情绪指数、最新金价
- **📰 新闻分析**: 浏览和分析新闻文章，支持情绪分析触发
- **📊 情绪趋势**: 不同时间段和类别的新闻情绪可视化
- **🔮 价格预测**: 基于机器学习模型的黄金价格预测
- **📊 分析洞察**: 情绪分析统计、数据质量指标

### 🚀 启动仪表板

#### 方法1: 专用启动脚本（推荐）
```bash
python start_dashboard.py
```

#### 方法2: 交互式启动
```bash
python run.py simple
# 选择选项2启动仪表板
```

#### 方法3: 直接启动
```bash
python -m streamlit run dashboard/app.py
```

#### 方法4: Windows批处理（双击运行）
```bash
start-dashboard.bat
```

### 🌐 访问地址

启动后，仪表板将在以下地址运行：

- ✅ `http://localhost:8501`
- ✅ `http://127.0.0.1:8501`
- ✅ `http://0.0.0.0:8501`

**💡 提示**: 如果某个地址无法访问，请尝试其他地址。

### 📱 使用说明

#### 1. 概览页面
- 查看系统关键指标
- 实时情绪指数图表
- 新闻来源分布
- 价格趋势图表

#### 2. 新闻分析页面
- 浏览最新的新闻文章
- 按类别筛选新闻
- 触发单个文章的情绪分析
- 快速操作按钮（收集新闻、更新情绪、清理数据）

#### 3. 情绪趋势页面
- 选择时间范围和新闻类别
- 查看情绪变化趋势
- 分析统计摘要

#### 4. 价格预测页面
- 查看当前预测结果
- 按模型类型筛选
- 特征重要性分析
- 触发模型训练和预测生成

#### 5. 分析页面
- 情绪分析洞察
- 系统性能指标
- 数据质量评估

### 🔧 故障排除

#### 无法访问仪表板
1. **确认仪表板正在运行**:
   ```bash
   python test_dashboard.py
   ```

2. **检查端口占用**:
   ```bash
   netstat -an | findstr :8501
   ```

3. **尝试不同URL**:
   - `http://localhost:8501`
   - `http://127.0.0.1:8501`
   - `http://0.0.0.0:8501`

4. **重启仪表板**:
   ```bash
   # 终止当前进程
   taskkill /PID <PID> /F
   # 重新启动
   python start_dashboard.py
   ```

#### 仪表板启动失败
1. **检查依赖**:
   ```bash
   pip install streamlit plotly altair
   ```

2. **检查文件权限**:
   ```bash
   python -c "import dashboard.app; print('Import OK')"
   ```

3. **查看详细错误**:
   ```bash
   python test_dashboard.py
   ```

### ⚙️ 配置说明

仪表板会自动检测可用的机器学习库：

- ✅ **VADER & TextBlob**: 基础情绪分析（总是可用）
- ⚠️ **FinBERT**: 需要 `transformers` 库
- ⚠️ **LSTM/XGBoost**: 需要 `tensorflow` 和 `xgboost` 库

### 📊 功能特性

#### 实时更新
- 仪表板会自动刷新数据
- 可以设置自动刷新间隔

#### 交互式图表
- 使用 Plotly 提供交互式图表
- 支持缩放、平移、悬停显示详细信息

#### 数据导出
- 支持 CSV、JSON 格式数据导出
- 报告生成功能

### 🔍 API集成

仪表板通过 RESTful API 与后端服务通信：

- 新闻数据: `/api/v1/news`
- 情绪数据: `/api/v1/sentiment`
- 预测数据: `/api/v1/predictions`
- 黄金价格: `/api/v1/gold-prices`

### 🛠️ 开发说明

#### 文件结构
```
dashboard/
├── app.py                 # 主仪表板应用
├── requirements.txt       # 仪表板依赖（可选）
└── README.md             # 仪表板文档
```

#### 自定义配置
可以通过修改 `dashboard/app.py` 来自定义：
- API 基础 URL
- 默认参数
- 图表样式
- 页面布局

### 📚 相关文档

- [完整系统文档](README.md)
- [故障排除指南](troubleshooting.md)
- [快速开始](QUICKSTART.md)

---

**🎉 仪表板已准备就绪！开始探索您的黄金价格预测数据吧！** 📊✨

# 🚀 快速开始指南

## 黄金价格预测模型的新闻情绪分析系统

### ✅ 系统已完全修复并可运行！

**🎉 所有功能都已确认正常工作！**

- ✅ API服务器
- ✅ 仪表板
- ✅ 数据库
- ✅ 机器学习模型
- ✅ 实时数据可视化

## 1. 安装依赖

```bash
pip install -r requirements-simple.txt
```

## 2. 创建数据库

```bash
python -c "from app.models.database import create_tables; create_tables()"
```

## 3. 启动服务

### 方法A: 交互式启动（推荐）
```bash
python run.py simple
```
选择要启动的服务：
1. API 服务器
2. 仪表板
3. 两者都启动

### 方法B: 直接启动
```bash
# 启动API服务器
python main.py

# 启动仪表板（新终端）
python -m streamlit run dashboard/app.py

# 或者使用专用启动脚本
python start_dashboard.py

# 或者双击Windows批处理文件
start-dashboard.bat
```

## 4. 访问服务

- API文档: http://localhost:8000/docs
- 仪表板: http://localhost:8501 (或 http://127.0.0.1:8501)
- 健康检查: http://localhost:8000/api/v1/health

## 5. 测试系统

```bash
# 基本系统测试
python test_startup.py

# API服务器测试
python test_api.py

# 仪表板测试
python test_dashboard.py
```

## 🎯 系统功能

### 确认工作的功能
- ✅ 新闻API (NewsAPI, Yahoo Finance)
- ✅ 情绪分析 (VADER, TextBlob)
- ✅ SQLite数据库
- ✅ RESTful API
- ✅ 数据可视化仪表板
- ✅ 完整的机器学习功能 (LSTM, XGBoost, FinBERT)

### 已安装的高级功能
系统已包含完整的机器学习功能：

- ✅ **FinBERT情绪分析** - 先进的金融文本情绪分析
- ✅ **LSTM价格预测** - 基于深度学习的黄金价格预测
- ✅ **XGBoost预测** - 梯度提升决策树预测模型
- ✅ **集成预测模型** - 组合多种模型的预测结果

## 故障排除

### 如果遇到导入错误
```bash
# 确保从项目目录运行
cd "F:\Financial Project\gold news"

# 重新安装依赖
pip install -r requirements-simple.txt

# 测试系统
python test_startup.py
```

### 如果API服务器无法启动
```bash
# 检查语法
python -c "import main; print('Syntax OK')"

# 运行完整测试
python test_api.py
```

### 如果仪表板无法启动
```bash
# 检查依赖
pip install streamlit plotly altair

# 测试仪表板
python test_dashboard.py

# 或者直接启动
python -m streamlit run dashboard/app.py

# 或者使用批处理文件（Windows）
start-dashboard.bat
```

## 更多信息

- 完整文档: [README.md](README.md)
- Windows指南: [README-Windows.md](README-Windows.md)
- 仪表板指南: [README-Dashboard.md](README-Dashboard.md)
- 故障排除: [troubleshooting.md](troubleshooting.md)
- 最终状态: [FINAL_STATUS.md](FINAL_STATUS.md)

---

**🎉 系统已完全修复并可运行！**

- ✅ API服务器正常运行
- ✅ 仪表板正常运行
- ✅ 数据库连接正常
- ✅ 所有API端点正常响应
- ✅ 数据可视化功能完整

**现在可以开始使用您的黄金价格预测系统了！**

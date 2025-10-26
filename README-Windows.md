# Windows 用户指南 - 黄金价格预测模型的新闻情绪分析系统

## 🪟 Windows 特殊说明

本指南专门为 Windows 用户提供简化的安装和使用说明，避免了复杂的 PostgreSQL 和 Redis 配置。

## 📋 系统要求

- Windows 10/11 (64位)
- Python 3.8+
- 至少 4GB RAM (推荐 8GB)
- 至少 2GB 可用磁盘空间

## 🚀 快速开始

### 方法1: 使用自动化脚本 (推荐)

1. **双击运行设置脚本**
```bash
setup-windows.bat
```

这个脚本会自动：
- 安装必要的依赖包
- 创建所需目录
- 设置SQLite数据库
- 显示使用说明

### 方法2: 手动安装

1. **安装Python依赖**
```bash
pip install -r requirements-simple.txt
```

2. **创建目录**
```bash
mkdir logs models data
```

3. **初始化数据库**
```bash
python -c "from app.models.database import create_tables; create_tables()"
```

## 🎯 启动服务

### 使用交互式启动器
```bash
python run.py simple
```
然后选择要启动的服务：
1. API 服务器
2. 仪表板
3. 两者都启动

### 单独启动服务

**启动API服务器:**
```bash
python run.py api
# 或直接运行
python main.py
```

**启动仪表板:**
```bash
python run.py dashboard
# 或直接运行
streamlit run dashboard/app.py
```

## 🌐 访问地址

- **API 文档**: http://localhost:8000/docs
- **监控仪表板**: http://localhost:8501
- **健康检查**: http://localhost:8000/api/v1/health

## 🔧 常见问题解决

### 1. pip install 失败

**问题**: `pip install -r requirements.txt` 安装失败

**解决方案**:
```bash
# 使用简化版本
pip install -r requirements-simple.txt

# 或使用Windows兼容版本
pip install -r requirements-windows.txt
```

### 2. 权限错误

**问题**: 安装时出现权限错误

**解决方案**:
```bash
# 使用用户安装模式
pip install --user -r requirements-simple.txt

# 或以管理员身份运行命令提示符
```

### 3. 缺少 Microsoft Visual C++ Build Tools

**问题**: 安装某些包时需要 C++ 编译器

**解决方案**:
1. 下载并安装 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. 或者使用预编译的包版本

### 4. 端口被占用

**问题**: 端口 8000 或 8501 被其他程序占用

**解决方案**:
```python
# 编辑配置文件修改端口
# app/core/config.py
API_PORT=8001  # 改为其他端口
DASHBOARD_PORT=8502
```

## 📦 依赖包说明

### 简化版本 (requirements-simple.txt)
- ✅ 基本功能完整
- ✅ 不需要 PostgreSQL/Redis
- ✅ 使用 SQLite 数据库
- ✅ 包含基本情绪分析 (VADER, TextBlob)
- ❌ 不包含高级模型 (FinBERT, LSTM, XGBoost)

### 完整版本 (requirements.txt)
- ✅ 所有功能完整
- ✅ 包含高级机器学习模型
- ❌ 需要 PostgreSQL 和 Redis
- ❌ 安装复杂度高

### Windows 兼容版本 (requirements-windows.txt)
- ✅ 针对 Windows 优化
- ✅ 兼容性更好
- ❌ 某些包可能需要手动安装

## 🛠️ 高级配置

### 1. 使用自己的API密钥

编辑配置文件添加API密钥：
```python
# 建议创建 .env 文件
NEWSAPI_KEY=your_newsapi_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here
```

### 2. 修改数据库路径

```python
# app/core/config.py
DATABASE_URL=sqlite:///./my_custom_path.db
```

### 3. 调整模型参数

```python
# app/core/config.py
BATCH_SIZE=8  # 减少批处理大小以降低内存使用
MAX_SEQUENCE_LENGTH=256  # 减少序列长度
```

## 📊 使用示例

### 1. 健康检查
```bash
curl http://localhost:8000/api/v1/health
```

### 2. 获取新闻
```bash
curl "http://localhost:8000/api/v1/news?limit=10"
```

### 3. 收集新闻数据
```bash
curl -X POST "http://localhost:8000/api/v1/news/collect"
```

### 4. 获取情绪数据
```bash
curl "http://localhost:8000/api/v1/sentiment?time_horizon=short"
```

## 🔍 故障排除

### 检查依赖是否正确安装
```bash
python run.py
# 这会检查并显示缺失的依赖
```

### 查看详细错误信息
```bash
# 检查日志文件
type logs\*.log
```

### 重置数据库
```bash
# 删除数据库文件重新创建
del gold_news.db
python -c "from app.models.database import create_tables; create_tables()"
```

### 更新依赖
```bash
pip install --upgrade -r requirements-simple.txt
```

## 💡 性能优化

1. **减少内存使用**: 降低 `BATCH_SIZE` 配置
2. **加快启动**: 使用 SQLite 而不是 PostgreSQL
3. **减少磁盘使用**: 定期清理旧数据

## 📚 学习资源

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [Streamlit 文档](https://docs.streamlit.io/)
- [SQLAlchemy 文档](https://docs.sqlalchemy.org/)

## 🆘 获取帮助

如果遇到问题：

1. 检查错误信息和日志
2. 尝试简化版本的依赖
3. 查看 README.md 获取完整说明
4. 提交 GitHub Issue

---

**注意**: Windows 版本针对易用性进行了优化，但功能完整性与完整版本相同。建议在熟悉系统后考虑升级到完整版本以获得更好的性能和功能。

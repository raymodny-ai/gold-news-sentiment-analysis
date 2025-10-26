# 黄金新闻情感分析系统 (Gold News Sentiment Analysis System)

一个基于机器学习的黄金新闻情感分析和价格预测系统,能够自动收集新闻、分析情感并预测黄金价格走势。

## 🌟 主要功能

- **📰 多源新闻采集**: 从NewsAPI、Yahoo Finance等多个来源自动采集黄金相关新闻
- **🎭 多模型情感分析**: 集成VADER、TextBlob、FinBERT等多种情感分析模型
- **📊 金价预测**: 使用LSTM、XGBoost等机器学习模型预测黄金价格
- **📈 可视化仪表板**: 基于Streamlit的交互式数据可视化仪表板
- **🔌 RESTful API**: 完整的FastAPI接口,支持所有核心功能

## 🏗️ 技术架构

### 后端技术栈
- **FastAPI**: 高性能异步Web框架
- **SQLAlchemy**: ORM数据库管理
- **SQLite**: 轻量级数据库
- **APScheduler**: 定时任务调度

### 机器学习
- **Transformers**: FinBERT金融情感分析
- **TensorFlow/Keras**: LSTM时序预测模型
- **XGBoost**: 梯度提升树预测
- **VADER & TextBlob**: 传统情感分析

### 前端展示
- **Streamlit**: 交互式仪表板
- **Plotly**: 动态图表可视化

## 📋 系统要求

- Python 3.8+
- 4GB+ RAM (推荐8GB用于ML模型)
- Windows/Linux/MacOS

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/gold-news-sentiment.git
cd gold-news-sentiment
```

### 2. 安装依赖

#### 完整安装 (包含所有ML功能)
```bash
pip install -r requirements.txt
```

#### 简化安装 (基础功能)
```bash
pip install -r requirements-simple.txt
```

### 3. 配置环境变量

创建 `.env` 文件:

```env
# API配置
API_HOST=127.0.0.1
API_PORT=8000
DEBUG=True

# 数据库
DATABASE_URL=sqlite:///./gold_news.db

# NewsAPI密钥 (可选)
NEWS_API_KEY=your_newsapi_key_here

# 日志级别
LOG_LEVEL=INFO
```

### 4. 初始化数据库

```bash
python -c "from app.models.database import create_tables; create_tables()"
```

### 5. 启动服务

#### 启动API服务器
```bash
python main.py
```

访问: http://localhost:8000
API文档: http://localhost:8000/docs

#### 启动仪表板
```bash
python start_dashboard.py
```
或
```bash
streamlit run dashboard/app.py
```

访问: http://localhost:8501

## 📁 项目结构

```
gold-news-sentiment/
├── app/                        # 应用核心代码
│   ├── api/                    # API路由
│   │   └── routes.py          # 所有API端点
│   ├── core/                   # 核心配置
│   │   └── config.py          # 系统配置
│   ├── models/                 # 数据模型
│   │   ├── database.py        # 数据库连接
│   │   └── schemas.py         # Pydantic模型
│   └── services/               # 业务服务
│       ├── news_collector.py  # 新闻采集
│       ├── sentiment_analyzer.py  # 情感分析
│       ├── prediction_model.py    # 价格预测
│       └── data_manager.py    # 数据管理
├── dashboard/                  # Streamlit仪表板
│   └── app.py                 # 仪表板主文件
├── models/                     # 保存的ML模型
├── logs/                       # 日志文件
├── main.py                     # API服务器入口
├── start_dashboard.py          # 仪表板启动脚本
├── requirements.txt            # 完整依赖
├── requirements-simple.txt     # 简化依赖
└── README.md                   # 项目文档
```

## 🔌 API端点

### 核心端点

- `GET /` - API根端点
- `GET /api/v1/health` - 健康检查
- `GET /api/info` - API信息

### 新闻相关

- `GET /api/v1/news` - 获取新闻列表
- `GET /api/v1/news/{news_id}` - 获取单条新闻
- `POST /api/v1/news/collect` - 手动触发新闻采集

### 情感分析

- `GET /api/v1/sentiment` - 获取情感分析结果
- `POST /api/v1/sentiment/analyze` - 分析指定新闻

### 价格预测

- `GET /api/v1/predictions` - 获取价格预测
- `POST /api/v1/predictions/train` - 训练预测模型
- `POST /api/v1/predictions/predict` - 生成新预测

### 金价数据

- `GET /api/v1/gold-prices` - 获取历史金价
- `POST /api/v1/gold-prices/fetch` - 更新金价数据

### 分析报告

- `GET /api/v1/analytics/sentiment-summary` - 情感分析摘要
- `GET /api/v1/analytics/price-trends` - 价格趋势分析
- `GET /api/v1/analytics/correlation` - 相关性分析

## 🎨 仪表板功能

1. **新闻监控**: 实时查看最新黄金新闻
2. **情感分析**: 可视化情感分布和趋势
3. **价格预测**: 查看AI预测的金价走势
4. **相关性分析**: 分析情感与价格的关系
5. **数据管理**: 手动触发数据收集和模型训练

## 🧪 测试

```bash
# 测试系统启动
python test_startup.py

# 测试API
python test_api.py

# 检查系统状态
python system_status.py
```

## 📊 数据源

- **NewsAPI**: 全球新闻数据
- **Yahoo Finance**: 金融新闻和金价数据
- **FRED API**: 美联储经济数据 (可选)

## 🤖 机器学习模型

### 情感分析模型

1. **VADER**: 规则基础情感分析
2. **TextBlob**: 传统NLP情感分析
3. **FinBERT**: 金融领域预训练BERT模型

### 价格预测模型

1. **LSTM**: 长短期记忆网络,用于时序预测
2. **XGBoost**: 梯度提升树,用于特征工程预测
3. **Ensemble**: 集成多个模型的预测结果

## 🛠️ 开发指南

### 添加新的数据源

在 `app/services/news_collector.py` 中添加新的采集器类:

```python
class NewSourceCollector(BaseCollector):
    async def collect_news(self, keywords, from_date, to_date):
        # 实现采集逻辑
        pass
```

### 自定义情感分析模型

在 `app/services/sentiment_analyzer.py` 中扩展:

```python
def custom_sentiment_analysis(self, text):
    # 实现自定义分析逻辑
    pass
```

## ⚠️ 常见问题

### 1. 导入错误
确保已安装所有依赖:
```bash
pip install -r requirements.txt
```

### 2. 数据库错误
重新创建数据库:
```bash
python -c "from app.models.database import create_tables; create_tables()"
```

### 3. 端口被占用
修改 `.env` 文件中的 `API_PORT` 或 `DASHBOARD_PORT`

### 4. ML模型加载失败
首次运行时模型会自动下载,确保网络连接正常

## 📝 更新日志

### v1.0.0 (2025-10-26)
- ✅ 完整的新闻采集系统
- ✅ 多模型情感分析
- ✅ LSTM/XGBoost价格预测
- ✅ Streamlit可视化仪表板
- ✅ RESTful API接口
- ✅ 自动化定时任务

## 🤝 贡献指南

欢迎提交Issue和Pull Request!

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 👥 作者

- 您的名字 - [GitHub主页](https://github.com/yourusername)

## 🙏 致谢

- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [NewsAPI](https://newsapi.org/)

## 📞 联系方式

- 项目主页: https://github.com/yourusername/gold-news-sentiment
- 问题反馈: https://github.com/yourusername/gold-news-sentiment/issues

---

⭐ 如果这个项目对您有帮助,请给个Star!

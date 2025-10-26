# 黄金价格预测模型的新闻情绪分析系统 - 技术栈

## 项目类型
基于机器学习的金融预测Web应用，包含新闻采集、情绪分析、价格预测和实时监控仪表板。

## 核心技术

### 主要编程语言
- **语言**: Python 3.8+
- **运行时**: CPython
- **包管理**: pip + requirements.txt, Poetry (可选)

### 关键依赖/库

#### 后端框架
- **FastAPI**: 高性能异步Web框架，用于构建RESTful API
- **Pydantic**: 数据验证和序列化
- **Uvicorn**: ASGI服务器，用于生产环境部署

#### 数据处理
- **pandas**: 数据分析和处理
- **numpy**: 数值计算
- **scikit-learn**: 机器学习工具包
- **SQLAlchemy**: ORM数据库操作
- **Alembic**: 数据库迁移管理

#### 机器学习
- **TensorFlow/PyTorch**: 深度学习框架（FinBERT模型）
- **transformers**: Hugging Face预训练模型库
- **XGBoost**: 梯度提升算法
- **LSTM**: 长短期记忆网络（通过TensorFlow/Keras）

#### 情绪分析
- **VADER**: 社交媒体情绪分析
- **TextBlob**: 简单情绪分析工具
- **FinBERT**: 金融领域预训练BERT模型
- **NLTK**: 自然语言处理工具包

#### 数据存储
- **PostgreSQL**: 主数据库
- **Redis**: 缓存和任务队列
- **psycopg2**: PostgreSQL适配器

#### 任务调度
- **Celery**: 分布式任务队列
- **Redis**: Celery消息代理

#### 前端界面
- **Streamlit**: 快速构建数据科学Web应用
- **Plotly**: 交互式图表库
- **Altair**: 统计可视化库

#### 外部API集成
- **requests**: HTTP客户端
- **yfinance**: Yahoo Finance数据获取
- **python-dotenv**: 环境变量管理

### 应用架构
**微服务架构模式**:
- 新闻采集服务 (News Collector)
- 情绪分析服务 (Sentiment Engine)  
- 预测模型服务 (Prediction Model)
- 数据存储层 (PostgreSQL + Redis)
- 监控仪表板 (Streamlit)

### 数据存储
- **主存储**: PostgreSQL 13+ (新闻数据、情绪分析结果、预测结果)
- **缓存**: Redis 6+ (情绪分析结果缓存、API响应缓存)
- **数据格式**: JSON (API响应), CSV (数据导出), Protocol Buffers (可选)

### 外部集成
- **APIs**: NewsAPI, Finnhub API, Yahoo Finance API
- **协议**: HTTP/REST, WebSocket (实时更新)
- **认证**: API密钥, OAuth (可选)

### 监控与仪表板技术
- **仪表板框架**: Streamlit
- **实时通信**: WebSocket, Server-Sent Events
- **可视化库**: Plotly, Altair, Matplotlib
- **状态管理**: Streamlit Session State

## 开发环境

### 构建与开发工具
- **构建系统**: Docker + Docker Compose
- **包管理**: pip, Poetry (可选)
- **开发工作流**: 热重载 (FastAPI), 文件监控 (Celery)

### 代码质量工具
- **静态分析**: pylint, flake8, mypy
- **格式化**: black, isort
- **测试框架**: pytest, pytest-asyncio
- **文档**: Sphinx, FastAPI自动文档

### 版本控制与协作
- **VCS**: Git
- **分支策略**: Git Flow
- **代码审查**: GitHub Pull Request

### 仪表板开发
- **热重载**: Streamlit自动重载
- **端口管理**: 可配置端口 (8501)
- **多实例支持**: 支持多个Streamlit应用同时运行

## 部署与分发

### 目标平台
- **云平台**: AWS, Azure, 阿里云
- **本地部署**: Docker容器
- **开发环境**: Windows, macOS, Linux

### 分发方法
- **Docker镜像**: 容器化部署
- **Python包**: pip安装
- **SaaS服务**: 云服务部署

### 安装要求
- **Python**: 3.8+
- **内存**: ≥8GB
- **存储**: ≥100GB
- **GPU**: 可选 (FinBERT模型加速)

### 更新机制
- **Docker镜像更新**: 滚动更新
- **数据库迁移**: Alembic版本控制
- **配置更新**: 环境变量

## 技术要求与约束

### 性能要求
- **API响应时间**: ≤200ms
- **情绪分析处理**: ≤5分钟/100条新闻
- **预测模型推理**: ≤30秒
- **仪表板加载**: ≤2秒
- **并发用户**: ≥100

### 兼容性要求
- **平台支持**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python版本**: 3.8-3.11
- **数据库版本**: PostgreSQL 12+
- **浏览器支持**: Chrome 90+, Firefox 88+, Safari 14+

### 安全与合规
- **安全要求**: HTTPS传输, API密钥管理, 数据加密
- **合规标准**: 金融数据保护, 用户隐私保护
- **威胁模型**: API攻击防护, 数据泄露防护, DDoS防护

### 可扩展性与可靠性
- **预期负载**: 1000+并发用户, 10万+新闻/天
- **可用性要求**: 99.5%正常运行时间
- **增长预测**: 支持10倍数据量增长

## 技术决策与理由

### 决策日志

1. **FastAPI选择**: 
   - 理由: 高性能异步框架, 自动API文档生成, 类型提示支持
   - 替代方案: Django, Flask
   - 权衡: 学习曲线 vs 性能优势

2. **PostgreSQL + Redis架构**:
   - 理由: PostgreSQL提供ACID保证, Redis提供高性能缓存
   - 替代方案: MongoDB, MySQL
   - 权衡: 复杂性 vs 数据一致性

3. **FinBERT模型选择**:
   - 理由: 金融领域预训练, 情绪分析准确率高
   - 替代方案: 通用BERT, RoBERTa
   - 权衡: 计算资源 vs 准确率

4. **Streamlit界面选择**:
   - 理由: 快速原型开发, 数据科学友好
   - 替代方案: React, Vue.js
   - 权衡: 开发速度 vs 定制化程度

5. **Celery任务队列**:
   - 理由: Python生态成熟, 支持分布式任务
   - 替代方案: RQ, Dramatiq
   - 权衡: 复杂性 vs 可靠性

## 已知限制

### 技术限制
- **FinBERT模型**: 需要GPU资源, 推理时间较长
- **API限制**: NewsAPI免费版100次/天限制
- **内存使用**: 大量新闻数据需要优化内存使用
- **并发处理**: 情绪分析任务可能成为瓶颈

### 业务限制
- **数据质量**: 依赖外部新闻源的数据质量
- **语言支持**: 主要支持英文和中文新闻
- **实时性**: 新闻采集有1小时延迟
- **准确性**: 情绪分析准确率约80%

### 未来改进
- **模型优化**: 使用更轻量级的情绪分析模型
- **缓存策略**: 优化缓存策略减少数据库压力
- **API扩展**: 集成更多新闻源提高覆盖率
- **性能优化**: 使用异步处理提高并发能力

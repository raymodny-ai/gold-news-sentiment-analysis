# 黄金价格预测模型的新闻情绪分析系统设计

## 系统架构设计

### 整体架构

系统采用微服务架构模式，包含以下核心组件：

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   新闻采集服务   │    │   情绪分析服务   │    │   预测模型服务   │
│  News Collector │───▶│ Sentiment Engine│───▶│ Prediction Model│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   数据存储层     │    │   缓存层        │    │   监控仪表板     │
│   PostgreSQL    │    │     Redis       │    │   Streamlit     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**数据流向**：
1. 新闻采集服务从多个API源获取新闻数据
2. 数据经过清洗和分类后存储到PostgreSQL
3. 情绪分析服务从数据库读取新闻，进行情绪分析
4. 分析结果存储到缓存层（Redis）和数据库
5. 预测模型服务结合历史价格数据和情绪指标进行预测
6. 监控仪表板实时展示情绪指数和预测结果

### 技术栈选择

**后端技术栈**：
- **编程语言**: Python 3.8+
- **Web框架**: FastAPI（高性能异步API）
- **任务队列**: Celery + Redis（异步任务处理）
- **数据库**: PostgreSQL 13+（主数据库）
- **缓存**: Redis 6+（情绪分析结果缓存）
- **机器学习**: scikit-learn, TensorFlow/PyTorch（FinBERT）

**前端技术栈**：
- **界面框架**: Streamlit（快速原型和仪表板）
- **可视化**: Plotly, Altair（交互式图表）
- **状态管理**: Streamlit Session State

**第三方服务集成**：
- **新闻API**: NewsAPI, Finnhub, Yahoo Finance
-
- **监控**: Prometheus + Grafana（可选）

### 模块设计

#### 1. 新闻采集模块 (news_collector)
```python
class NewsCollector:
    - collect_from_newsapi()
    - collect_from_finnhub()
    - collect_from_yahoo()
    - deduplicate_news()
    - classify_news()
```

#### 2. 情绪分析模块 (sentiment_analyzer)
```python
class SentimentAnalyzer:
    - analyze_with_vader()
    - analyze_with_textblob()
    - analyze_with_finbert()
    - calculate_sentiment_scores()
    - apply_time_decay()
```

#### 3. 预测模型模块 (prediction_model)
```python
class PredictionModel:
    - prepare_features()
    - train_lstm_model()
    - train_xgboost_model()
    - predict_price()
    - calculate_confidence()
```

#### 4. 数据管理模块 (data_manager)
```python
class DataManager:
    - store_news()
    - store_sentiment()
    - store_predictions()
    - get_historical_data()
    - cleanup_old_data()
```

## 数据库设计

### 数据表结构

#### 1. 新闻表 (news)
```sql
CREATE TABLE news (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50) NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    url VARCHAR(500),
    published_at TIMESTAMP NOT NULL,
    category VARCHAR(50) NOT NULL, -- 5类新闻分类
    language VARCHAR(10) DEFAULT 'en',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_news_published_at ON news(published_at);
CREATE INDEX idx_news_category ON news(category);
CREATE INDEX idx_news_source ON news(source);
```

#### 2. 情绪分析结果表 (sentiment_analysis)
```sql
CREATE TABLE sentiment_analysis (
    id SERIAL PRIMARY KEY,
    news_id INTEGER REFERENCES news(id),
    analyzer_type VARCHAR(20) NOT NULL, -- vader, textblob, finbert
    bullish_score DECIMAL(5,4), -- 看涨指数
    bearish_score DECIMAL(5,4), -- 看跌指数
    attention_score DECIMAL(5,4), -- 关注度指标
    confidence DECIMAL(5,4), -- 置信度
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sentiment_news_id ON sentiment_analysis(news_id);
CREATE INDEX idx_sentiment_analyzer ON sentiment_analysis(analyzer_type);
```

#### 3. 加权情绪指数表 (weighted_sentiment)
```sql
CREATE TABLE weighted_sentiment (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    category VARCHAR(50) NOT NULL,
    weighted_score DECIMAL(8,6) NOT NULL,
    time_horizon VARCHAR(20) NOT NULL, -- short, medium, long
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX idx_weighted_sentiment_unique ON weighted_sentiment(date, category, time_horizon);
```

#### 4. 价格预测结果表 (price_predictions)
```sql
CREATE TABLE price_predictions (
    id SERIAL PRIMARY KEY,
    prediction_date DATE NOT NULL,
    target_date DATE NOT NULL,
    model_type VARCHAR(20) NOT NULL, -- lstm, xgboost, ensemble
    predicted_price DECIMAL(10,2) NOT NULL,
    confidence_interval_lower DECIMAL(10,2),
    confidence_interval_upper DECIMAL(10,2),
    feature_importance JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_date ON price_predictions(prediction_date);
CREATE INDEX idx_predictions_target ON price_predictions(target_date);
```

#### 5. 黄金价格历史表 (gold_prices)
```sql
CREATE TABLE gold_prices (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    open_price DECIMAL(10,2),
    high_price DECIMAL(10,2),
    low_price DECIMAL(10,2),
    close_price DECIMAL(10,2),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_gold_prices_date ON gold_prices(date);
```

### 数据存储策略

**数据分区策略**：
- 按月份分区存储新闻数据
- 按年份分区存储价格历史数据
- 自动清理6个月前的原始新闻数据

**备份和恢复方案**：
- 每日自动备份数据库
- 保留30天的备份文件
- 支持点时间恢复

## API设计

### RESTful API

#### 1. 新闻相关API
```python
# 获取新闻列表
GET /api/v1/news
Query Parameters:
- category: 新闻类别
- start_date: 开始日期
- end_date: 结束日期
- limit: 限制数量
- offset: 偏移量

Response:
{
    "data": [
        {
            "id": 1,
            "title": "美联储降息预期升温",
            "content": "...",
            "category": "macro_policy",
            "published_at": "2025-10-23T10:00:00Z",
            "source": "newsapi"
        }
    ],
    "total": 100,
    "page": 1,
    "per_page": 20
}
```

#### 2. 情绪分析API
```python
# 获取情绪指数
GET /api/v1/sentiment
Query Parameters:
- category: 新闻类别
- time_horizon: 时间范围 (short/medium/long)
- start_date: 开始日期
- end_date: 结束日期

Response:
{
    "data": [
        {
            "date": "2025-10-23",
            "category": "macro_policy",
            "weighted_score": 0.75,
            "bullish_score": 0.8,
            "bearish_score": 0.2,
            "attention_score": 0.9
        }
    ]
}
```

#### 3. 预测结果API
```python
# 获取价格预测
GET /api/v1/predictions
Query Parameters:
- target_date: 预测目标日期
- model_type: 模型类型

Response:
{
    "data": {
        "prediction_date": "2025-10-23",
        "target_date": "2025-10-24",
        "predicted_price": 2650.50,
        "confidence_interval": {
            "lower": 2600.00,
            "upper": 2700.00
        },
        "model_type": "ensemble",
        "feature_importance": {
            "sentiment_score": 0.35,
            "technical_indicators": 0.25,
            "economic_data": 0.40
        }
    }
}
```

### 数据格式

**JSON Schema定义**：
```json
{
    "NewsSchema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "maxLength": 500},
            "content": {"type": "string"},
            "category": {"type": "string", "enum": ["macro_policy", "geopolitical", "economic_data", "market_sentiment", "central_bank"]},
            "published_at": {"type": "string", "format": "date-time"},
            "source": {"type": "string", "enum": ["newsapi", "finnhub", "yahoo"]}
        },
        "required": ["title", "category", "published_at", "source"]
    }
}
```

## 安全设计

### 认证和授权

**API密钥管理**：
- 使用环境变量存储API密钥
- 实现密钥轮换机制
- 支持多环境密钥配置

**访问控制**：
- API访问频率限制（每分钟100次）
- IP白名单机制
- 用户角色权限控制

### 数据安全

**数据加密**：
- 传输层使用HTTPS/TLS 1.3
- 数据库连接使用SSL
- 敏感数据字段加密存储

**数据保护**：
- 定期安全审计
- 数据访问日志记录
- 个人信息脱敏处理

## 性能设计

### 性能指标

**响应时间要求**：
- API响应时间 ≤ 200ms
- 情绪分析处理时间 ≤ 5分钟/100条新闻
- 预测模型推理时间 ≤ 30秒
- 仪表板页面加载时间 ≤ 2秒

**吞吐量要求**：
- 支持并发用户数 ≥ 100
- 新闻处理能力 ≥ 1000条/小时
- 数据库查询QPS ≥ 1000

### 优化策略

**缓存策略**：
- Redis缓存情绪分析结果（TTL: 1小时）
- 缓存API响应结果（TTL: 5分钟）
- 缓存预测结果（TTL: 30分钟）

**数据库优化**：
- 合理设计索引
- 查询优化和慢查询监控
- 连接池管理
- 读写分离（可选）

**代码优化**：
- 异步处理新闻采集
- 批量处理情绪分析
- 模型预测结果缓存
- 内存使用优化

## 部署设计

### 环境配置


**生产环境**：
- 使用Kubernetes部署
- 配置负载均衡
- 自动扩缩容
- 健康检查和监控

### 部署策略



COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**CI/CD流程**：
1. 代码提交触发构建
2. 运行单元测试和集成测试
3. 构建Docker镜像
4. 部署到测试环境
5. 运行端到端测试
6. 部署到生产环境

## 测试设计

### 测试策略

**单元测试**：
- 新闻采集模块测试
- 情绪分析算法测试
- 预测模型测试
- API端点测试

**集成测试**：
- 数据库集成测试
- 外部API集成测试
- 缓存系统测试
- 消息队列测试

**端到端测试**：
- 完整业务流程测试
- 用户界面测试
- 性能测试
- 安全测试

### 测试数据

**测试数据准备**：
- 模拟新闻数据（1000条）
- 历史价格数据（3年）
- 情绪分析结果数据
- 预测结果数据

**性能测试数据**：
- 大规模新闻数据（10万条）
- 高并发访问模拟
- 长时间运行测试
- 内存泄漏检测

## 监控和运维

### 系统监控

**应用监控**：
- API响应时间监控
- 错误率监控
- 数据库性能监控
- 缓存命中率监控

**业务监控**：
- 新闻采集成功率
- 情绪分析准确率
- 预测模型性能
- 用户访问统计

### 日志管理

**日志级别**：
- ERROR: 系统错误
- WARN: 警告信息
- INFO: 一般信息
- DEBUG: 调试信息

**日志存储**：
- 本地文件存储
- 集中式日志收集（ELK Stack）
- 日志轮转和清理
- 敏感信息脱敏

这个设计文档完全基于需求文档和指南要求，提供了完整的技术实现方案。



"""
Celery settings configuration.
"""
from app.core.config import settings

# Broker settings
broker_url = settings.celery_broker_url
result_backend = settings.celery_result_backend

# Task settings
task_serializer = 'json'
accept_content = ['json']
result_serializer = 'json'
timezone = 'UTC'
enable_utc = True

# Worker settings
worker_prefetch_multiplier = 1
worker_max_tasks_per_child = 1000
worker_disable_rate_limits = False

# Task routing
task_routes = {
    'app.tasks.news_tasks.*': {'queue': 'news'},
    'app.tasks.sentiment_tasks.*': {'queue': 'sentiment'},
    'app.tasks.prediction_tasks.*': {'queue': 'prediction'},
    'app.tasks.cleanup_tasks.*': {'queue': 'cleanup'},
}

# Task execution settings
task_acks_late = True
task_reject_on_worker_lost = True

# Result expiration
result_expires = 3600  # 1 hour

# Logging
worker_log_color = False
worker_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
worker_task_log_format = '[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s'

# Beat scheduler settings (for periodic tasks)
beat_schedule = {
    'collect-news-every-hour': {
        'task': 'app.tasks.news_tasks.collect_news_periodically',
        'schedule': 3600.0,  # Every hour
    },
    'update-sentiment-indices-daily': {
        'task': 'app.tasks.sentiment_tasks.update_sentiment_indices_daily',
        'schedule': 86400.0,  # Daily
    },
    'train-models-weekly': {
        'task': 'app.tasks.prediction_tasks.train_models_weekly',
        'schedule': 604800.0,  # Weekly
    },
    'cleanup-old-data-weekly': {
        'task': 'app.tasks.cleanup_tasks.cleanup_old_data_weekly',
        'schedule': 604800.0,  # Weekly
    },
}

# Security settings
worker_disable_rate_limits = False
task_always_eager = False
task_eager_propagates = False

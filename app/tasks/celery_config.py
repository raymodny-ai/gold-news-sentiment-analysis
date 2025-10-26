"""
Celery configuration for background task processing.
"""
import os
from celery import Celery
from app.core.config import settings

# Set default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gold_news.settings')

# Create Celery app
celery_app = Celery('gold_news')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
celery_app.config_from_object('app.tasks.celery_settings')

# Load task modules from all registered Django app configs.
celery_app.autodiscover_tasks()


@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery."""
    print(f'Request: {self.request!r}')

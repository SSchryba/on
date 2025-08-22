"""Gunicorn configuration for production."""

import multiprocessing

workers = int(multiprocessing.cpu_count() * 2 + 1)
bind = "0.0.0.0:8000"
timeout = 60
graceful_timeout = 30
keepalive = 5
worker_class = "uvicorn.workers.UvicornWorker"
loglevel = "info"
accesslog = "-"
errorlog = "-"

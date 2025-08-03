web: gunicorn main:app \
      --preload \
      --workers ${WEB_CONCURRENCY:-2} \
      --threads 4 \
      --timeout 300 \            # 5-minute worker timeout
      --log-file -

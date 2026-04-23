FROM docker.io/vllm/vllm-openai:nightly

RUN pip install --no-cache-dir nvidia-ml-py \
    && pip install --no-cache-dir "transformers>=4.56,<5"

COPY admin/ /app/admin/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /app

EXPOSE 8001-8010 7080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7080/healthz', timeout=3).status == 200 else 1)" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]

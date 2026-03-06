FROM docker.io/vllm/vllm-openai:v0.11.2

RUN pip install --no-cache-dir nvidia-ml-py

COPY patches/ /tmp/patches/
RUN python3 /tmp/patches/fix_qwen35_moe.py || echo "Patch not needed"; rm -rf /tmp/patches

COPY admin/ /app/admin/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /app

EXPOSE 8001-8010 7080

ENTRYPOINT ["/app/entrypoint.sh"]

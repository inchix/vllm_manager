#!/bin/bash
exec uvicorn admin.app:app --host 0.0.0.0 --port 7080

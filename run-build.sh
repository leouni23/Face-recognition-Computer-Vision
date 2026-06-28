#!/usr/bin/env bash
cd /home/leo/Face-recognition-Computer-Vision
echo "BUILD_START $(date)"
docker compose -f docker-compose.jetson.yml --env-file .env.jetson up --build -d 2>&1
echo "BUILD_EXIT $? $(date)"

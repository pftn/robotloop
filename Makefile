.PHONY: help up down restart build clean logs ps

# 默认目标
help:
	@echo "RobotLoop 机器人多模态数据闭环平台 Demo"
	@echo ""
	@echo "可用命令:"
	@echo "  make up         启动所有服务"
	@echo "  make down       停止并删除所有服务"
	@echo "  make restart    重启所有服务"
	@echo "  make build      构建所有镜像"
	@echo "  make clean      停止服务并删除数据卷"
	@echo "  make logs       查看所有服务日志"
	@echo "  make ps         查看服务状态"

up:
	docker-compose up -d

down:
	docker-compose down

restart:
	docker-compose down && docker-compose up -d

build:
	docker-compose build --no-cache

clean:
	docker-compose down -v

logs:
	docker-compose logs -f

ps:
	docker-compose ps
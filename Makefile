.PHONY: help build up down restart logs clean clean-all test shell env status health ps

# Variables
DOCKER_COMPOSE = docker-compose
DOCKER = docker
APP_CONTAINER = elysia-app
WEAVIATE_CONTAINER = elysia-weaviate
OLLAMA_CONTAINER = elysia-ollama

# Colors for output
BLUE = \033[0;34m
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
NC = \033[0m # No Color

##@ General

help: ## Display this help message
	@echo "$(BLUE)Elysia - Docker Management$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

setup: ## Initial setup - create .env from .env.example
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Creating .env file from .env.example...$(NC)"; \
		cp .env.example .env; \
		echo "$(GREEN)✓ .env file created. Please edit it with your API keys.$(NC)"; \
	else \
		echo "$(YELLOW)⚠ .env file already exists. Skipping...$(NC)"; \
	fi

env: setup ## Alias for setup

##@ Docker Operations

build: ## Build all Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	@$(DOCKER_COMPOSE) build
	@echo "$(GREEN)✓ Build complete!$(NC)"

up: ## Start all services (Elysia + Weaviate)
	@echo "$(BLUE)Starting all services...$(NC)"
	@$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✓ Services started!$(NC)"
	@echo ""
	@echo "$(GREEN)Elysia is starting up...$(NC)"
	@echo "$(YELLOW)Please wait 30-40 seconds for the application to be ready.$(NC)"
	@echo ""
	@echo "Access the application at: $(BLUE)http://localhost:8000$(NC)"
	@echo "Weaviate is available at: $(BLUE)http://localhost:8080$(NC)"
	@echo ""
	@echo "Run '$(GREEN)make logs$(NC)' to see the logs"
	@echo "Run '$(GREEN)make health$(NC)' to check service health"

up-ollama: ## Start all services including Ollama (for local LLM)
	@echo "$(BLUE)Starting all services with Ollama...$(NC)"
	@$(DOCKER_COMPOSE) --profile local-llm up -d
	@echo "$(GREEN)✓ Services started with Ollama!$(NC)"
	@echo ""
	@echo "Ollama is available at: $(BLUE)http://localhost:11434$(NC)"
	@echo "To pull a model: $(GREEN)make ollama-pull MODEL=llama2$(NC)"

down: ## Stop all services
	@echo "$(BLUE)Stopping all services...$(NC)"
	@$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ Services stopped!$(NC)"

restart: down up ## Restart all services

##@ Application Management

start: up ## Start the application (alias for up)

stop: down ## Stop the application (alias for down)

run: build up ## Build and start the application
	@echo "$(GREEN)✓ Application is running!$(NC)"

##@ Logs and Monitoring

logs: ## Show logs from all services
	@$(DOCKER_COMPOSE) logs -f

logs-app: ## Show logs from Elysia application only
	@$(DOCKER_COMPOSE) logs -f $(APP_CONTAINER)

logs-weaviate: ## Show logs from Weaviate only
	@$(DOCKER_COMPOSE) logs -f $(WEAVIATE_CONTAINER)

logs-ollama: ## Show logs from Ollama only
	@$(DOCKER_COMPOSE) logs -f $(OLLAMA_CONTAINER)

ps: ## Show running containers
	@$(DOCKER_COMPOSE) ps

status: ps ## Show status of services (alias for ps)

health: ## Check health of all services
	@echo "$(BLUE)Checking service health...$(NC)"
	@echo ""
	@echo "$(YELLOW)Weaviate:$(NC)"
	@curl -sf http://localhost:8080/v1/.well-known/ready > /dev/null && echo "  $(GREEN)✓ Healthy$(NC)" || echo "  $(RED)✗ Not responding$(NC)"
	@echo "$(YELLOW)Elysia API:$(NC)"
	@curl -sf http://localhost:8000/api/health > /dev/null && echo "  $(GREEN)✓ Healthy$(NC)" || echo "  $(RED)✗ Not responding$(NC)"

##@ Development

shell: ## Open a shell in the Elysia container
	@$(DOCKER) exec -it $(APP_CONTAINER) /bin/bash

shell-weaviate: ## Open a shell in the Weaviate container
	@$(DOCKER) exec -it $(WEAVIATE_CONTAINER) /bin/sh

dev: ## Start in development mode with live reload
	@echo "$(BLUE)Starting in development mode...$(NC)"
	@$(DOCKER_COMPOSE) up

rebuild: ## Rebuild and restart the application
	@echo "$(BLUE)Rebuilding application...$(NC)"
	@$(DOCKER_COMPOSE) build --no-cache $(APP_CONTAINER)
	@$(DOCKER_COMPOSE) up -d $(APP_CONTAINER)
	@echo "$(GREEN)✓ Application rebuilt and restarted!$(NC)"

build-frontend: ## Build frontend and export to static_custom
	@echo "$(BLUE)Building frontend...$(NC)"
	@cd frontend && npm install && npm run build
	@echo "$(GREEN)✓ Frontend built and exported to elysia/api/static_custom/$(NC)"

##@ Ollama Management

ollama-pull: ## Pull a model for Ollama (usage: make ollama-pull MODEL=llama2)
	@if [ -z "$(MODEL)" ]; then \
		echo "$(RED)Error: MODEL not specified. Usage: make ollama-pull MODEL=llama2$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Pulling Ollama model: $(MODEL)...$(NC)"
	@$(DOCKER) exec $(OLLAMA_CONTAINER) ollama pull $(MODEL)
	@echo "$(GREEN)✓ Model $(MODEL) pulled successfully!$(NC)"

ollama-list: ## List available Ollama models
	@$(DOCKER) exec $(OLLAMA_CONTAINER) ollama list

ollama-run: ## Run a model interactively (usage: make ollama-run MODEL=llama2)
	@if [ -z "$(MODEL)" ]; then \
		echo "$(RED)Error: MODEL not specified. Usage: make ollama-run MODEL=llama2$(NC)"; \
		exit 1; \
	fi
	@$(DOCKER) exec -it $(OLLAMA_CONTAINER) ollama run $(MODEL)

##@ Cleanup

clean: ## Remove stopped containers and networks
	@echo "$(BLUE)Cleaning up...$(NC)"
	@$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ Cleanup complete!$(NC)"

clean-volumes: ## Remove containers, networks, and volumes (WARNING: deletes data)
	@echo "$(RED)WARNING: This will delete all data in Weaviate and Ollama!$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to cancel, or wait 5 seconds to continue...$(NC)"
	@sleep 5
	@$(DOCKER_COMPOSE) down -v
	@echo "$(GREEN)✓ Volumes removed!$(NC)"

clean-all: ## Remove everything including images (WARNING: full cleanup)
	@echo "$(RED)WARNING: This will remove all containers, volumes, and images!$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to cancel, or wait 5 seconds to continue...$(NC)"
	@sleep 5
	@$(DOCKER_COMPOSE) down -v --rmi all
	@echo "$(GREEN)✓ Full cleanup complete!$(NC)"

clean-logs: ## Clean application logs
	@echo "$(BLUE)Cleaning logs...$(NC)"
	@rm -rf logs/*
	@echo "$(GREEN)✓ Logs cleaned!$(NC)"

##@ Testing

test: ## Run tests in the container
	@echo "$(BLUE)Running tests...$(NC)"
	@$(DOCKER) exec $(APP_CONTAINER) pytest tests/

test-no-reqs: ## Run tests that don't require external services
	@echo "$(BLUE)Running tests (no requirements)...$(NC)"
	@$(DOCKER) exec $(APP_CONTAINER) pytest tests/no_reqs/

test-requires-env: ## Run tests that require environment setup
	@echo "$(BLUE)Running tests (requires environment)...$(NC)"
	@$(DOCKER) exec $(APP_CONTAINER) pytest tests/requires_env/

##@ Maintenance

update: ## Update dependencies and rebuild
	@echo "$(BLUE)Updating dependencies...$(NC)"
	@$(DOCKER_COMPOSE) build --no-cache
	@$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✓ Update complete!$(NC)"

backup-data: ## Backup Weaviate data
	@echo "$(BLUE)Creating backup of Weaviate data...$(NC)"
	@mkdir -p backups
	@$(DOCKER) exec $(WEAVIATE_CONTAINER) tar czf /tmp/weaviate-backup.tar.gz /var/lib/weaviate
	@$(DOCKER) cp $(WEAVIATE_CONTAINER):/tmp/weaviate-backup.tar.gz backups/weaviate-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz
	@echo "$(GREEN)✓ Backup created in backups/ directory$(NC)"

##@ Information

info: ## Show system information
	@echo "$(BLUE)System Information$(NC)"
	@echo ""
	@echo "$(YELLOW)Docker Version:$(NC)"
	@$(DOCKER) --version
	@echo ""
	@echo "$(YELLOW)Docker Compose Version:$(NC)"
	@$(DOCKER_COMPOSE) version
	@echo ""
	@echo "$(YELLOW)Running Containers:$(NC)"
	@$(DOCKER_COMPOSE) ps
	@echo ""
	@echo "$(YELLOW)Disk Usage:$(NC)"
	@$(DOCKER) system df

ports: ## Show exposed ports
	@echo "$(BLUE)Service Ports$(NC)"
	@echo ""
	@echo "$(GREEN)Elysia API:$(NC)       http://localhost:8000"
	@echo "$(GREEN)Weaviate:$(NC)         http://localhost:8080"
	@echo "$(GREEN)Weaviate gRPC:$(NC)    localhost:50051"
	@echo "$(GREEN)Ollama (optional):$(NC) http://localhost:11434"

docs: ## Show link to documentation
	@echo "$(BLUE)Documentation$(NC)"
	@echo ""
	@echo "Online docs: $(GREEN)https://weaviate.github.io/elysia/$(NC)"
	@echo "Local docs:  $(GREEN)file://$(shell pwd)/docs/index.md$(NC)"
	@echo ""
	@echo "Docker setup: $(GREEN)file://$(shell pwd)/DOCKER.md$(NC)"

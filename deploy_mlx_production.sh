#!/bin/bash
# MLX 生产环境部署脚本

echo "🚀 MLX 生产环境部署"
echo "=================="

# 检查 Python 环境
echo "🔍 检查 Python 环境..."
python --version

# 检查依赖
echo "🔍 检查依赖..."
python -c "import mlx.core; print('✅ MLX 可用')" || echo "❌ MLX 不可用"
python -c "import torch; print('✅ PyTorch 可用')" || echo "❌ PyTorch 不可用"

# 运行环境检查
echo "🔍 运行环境检查..."
python -c "
from apply_production_fixes import check_production_requirements
check_production_requirements()
"

# 部署模型
echo "🚀 部署模型..."
python -c "
from apply_production_fixes import deploy_mlx_model
deploy_mlx_model()
"

# 运行测试
echo "🧪 运行测试..."
python automated_tests.py

echo "✅ 部署完成!"

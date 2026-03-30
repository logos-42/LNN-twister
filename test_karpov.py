"""
卡帕斯循环系统快速测试脚本
"""

print("="*70)
print("卡帕斯循环系统 - 快速测试")
print("="*70)

# 测试 1: 检查 Ollama 连接
print("\n测试 1: 检查 Ollama 连接")
print("-"*40)

try:
    import requests
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    
    if response.status_code == 200:
        models = response.json().get('models', [])
        print(f"✓ Ollama 连接成功")
        print(f"  可用模型：{[m['name'] for m in models]}")
    else:
        print(f"⚠ Ollama 返回状态码：{response.status_code}")
except requests.exceptions.ConnectionError:
    print("✗ 无法连接到 Ollama")
    print("\n请执行以下步骤:")
    print("  1. 安装 Ollama: https://ollama.ai")
    print("  2. 启动 Ollama: ollama serve")
    print("  3. 下载模型：ollama pull llama2")
except Exception as e:
    print(f"✗ 错误：{e}")

# 测试 2: 检查代码分析器
print("\n测试 2: 检查代码分析器")
print("-"*40)

try:
    from karpov_cycle import CodeAnalyzer
    
    analyzer = CodeAnalyzer()
    stats = analyzer.get_file_stats()
    
    print(f"✓ 代码分析器正常")
    print(f"  文件数：{stats['total_files']}")
    print(f"  总行数：{stats['total_lines']}")
except Exception as e:
    print(f"✗ 代码分析器错误：{e}")

# 测试 3: 检查测试验证器
print("\n测试 3: 检查测试验证器")
print("-"*40)

try:
    from karpov_cycle import TestValidator
    
    validator = TestValidator()
    passed, message = validator.run_tests()
    
    if passed:
        print(f"✓ 测试验证器正常")
        print(f"  {message}")
    else:
        print(f"⚠ 测试验证器警告：{message}")
except Exception as e:
    print(f"✗ 测试验证器错误：{e}")

# 测试 4: 检查卡帕斯循环系统
print("\n测试 4: 检查卡帕斯循环系统")
print("-"*40)

try:
    from karpov_cycle import KarpovCycle, OllamaClient
    
    print(f"✓ 卡帕斯循环系统导入成功")
    print(f"\n使用方式:")
    print(f"  python karpov_cycle.py --iterations 3")
    print(f"  python karpov_cycle.py --model codellama")
    print(f"  python karpov_cycle.py --analyze-only")
except Exception as e:
    print(f"✗ 卡帕斯循环系统错误：{e}")

# 总结
print("\n" + "="*70)
print("快速测试完成")
print("="*70)

print("""
下一步:
  1. 确保 Ollama 正在运行：ollama serve
  2. 确保已下载模型：ollama pull llama2
  3. 运行卡帕斯循环：python karpov_cycle.py --iterations 3
  4. 查看使用指南：docs/卡帕斯循环系统指南.md
""")

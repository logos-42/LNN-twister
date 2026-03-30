# Twistor-LNN GitHub 自动存档指南

**版本**: 0.4.1  
**完成日期**: 2026-03-28  
**状态**: ✅ 自动存档系统完成

---

## 📋 概述

自动存档系统提供：
- ✅ 自动测试验证
- ✅ 自动提交到 Git
- ✅ 测试通过则推送 (存档)
- ✅ 测试失败则回滚
- ✅ GitHub Actions 集成

---

## 🚀 快速开始

### 本地存档

```bash
# 基本用法 (本地提交)
python auto_archive.py -m "v0.4.0 完成"

# 运行测试后提交
python auto_archive.py -t -m "v0.4.0 完成"

# 运行测试并推送到 GitHub
python auto_archive.py -t -p -m "v0.4.0 完成"

# 查看状态
python auto_archive.py --status

# 回滚
python auto_archive.py --rollback
```

### GitHub Actions 自动存档

推送到 main/master 分支时自动触发：

```yaml
# 自动运行测试并存档
push → GitHub Actions → 测试 → 通过 → 存档
                              ↓
                           失败 → 回滚
```

---

## 📁 文件结构

```
LNN-twister/
├── auto_archive.py                 # 自动存档脚本
├── archive_log.json                # 存档日志 (自动生成)
│
├── .github/
│   └── workflows/
│       └── auto-archive.yml        # GitHub Actions 工作流
│
├── .git_backup/                    # 备份目录 (临时)
│
└── docs/
    └── GitHub 自动存档指南.md       # 本文档
```

---

## 🔧 命令行参数

### auto_archive.py 参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `-m, --message` | 提交信息 | `-m "v0.4.0 完成"` |
| `-t, --test` | 运行测试 | `-t` |
| `-p, --push` | 自动推送 | `-p` |
| `--rollback` | 回滚 | `--rollback` |
| `--status` | 显示状态 | `--status` |
| `--no-test` | 跳过测试 | `--no-test` |
| `--no-push` | 不推送 | `--no-push` |

### 使用示例

```bash
# 1. 仅本地提交 (不测试，不推送)
python auto_archive.py -m "更新代码"

# 2. 测试后本地提交
python auto_archive.py -t -m "更新代码"

# 3. 测试后提交并推送
python auto_archive.py -t -p -m "v0.4.0 完成"

# 4. 跳过测试直接推送
python auto_archive.py -p -m "紧急修复" --no-test

# 5. 查看状态
python auto_archive.py --status

# 6. 回滚到上一版本
python auto_archive.py --rollback
```

---

## 📊 存档流程

### 完整流程

```
┌─────────────┐
│  开始存档   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  1. 创建备份 │ ← .git_backup
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  2. 运行测试 │ ← 如果失败则回滚
└──────┬──────┘
       │ ✓
       ▼
┌─────────────┐
│  3. 暂存更改 │ ← git add -A
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  4. 提交     │ ← git commit -m "..."
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  5. 推送     │ ← git push (可选)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  6. 清理备份 │ ← 删除 .git_backup
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  7. 记录日志 │ ← archive_log.json
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  存档完成   │
└─────────────┘
```

### 回滚流程

```
┌─────────────┐
│  开始回滚   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  检查备份   │
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
 有备份   无备份
   │       │
   ▼       ▼
恢复备份  git reset
   │       │
   └───┬───┘
       │
       ▼
┌─────────────┐
│  回滚完成   │
└─────────────┘
```

---

## 🧪 测试系统

### 内置测试

| 测试 | 说明 | 权重 |
|------|------|------|
| **import_test** | 导入和基础功能测试 | 高 |
| **evolution_test** | 进化系统测试 | 中 |
| **file_integrity_test** | 文件完整性测试 | 高 |

### 添加自定义测试

```python
# 在 auto_archive.py 中添加
def run_tests(self):
    # ... 现有测试 ...
    
    # 添加新测试
    print("  测试 4: 自定义测试...")
    try:
        # 你的测试逻辑
        from my_module import my_function
        result = my_function()
        assert result == expected
        
        print("    ✓ 自定义测试通过")
        test_results.append(("custom_test", True))
    except Exception as e:
        print(f"    ✗ 自定义测试失败：{e}")
        test_results.append(("custom_test", False))
```

---

## 📝 存档日志

### 日志格式

```json
{
  "logs": [
    {
      "timestamp": "2026-03-28T10:30:00",
      "message": "v0.4.0 完成",
      "commit": "abc1234567890",
      "test_passed": true,
      "pushed": true
    }
  ]
}
```

### 查看日志

```bash
# 查看日志文件
cat archive_log.json

# 或使用 Python
python -c "
import json
with open('archive_log.json') as f:
    logs = json.load(f)
for log in logs[-5:]:
    print(f\"{log['timestamp'][:10]}: {log['message']}\")
"
```

---

## 🔄 GitHub Actions 集成

### 触发条件

| 事件 | 说明 |
|------|------|
| **push** | 推送到 main/master/develop |
| **pull_request** | PR 到 main/master |
| **workflow_dispatch** | 手动触发 |

### 手动触发

1. 进入 GitHub 仓库
2. 点击 **Actions** 标签
3. 选择 **Auto Archive** 工作流
4. 点击 **Run workflow**
5. 填写参数：
   - `message`: 提交信息
   - `run_tests`: 是否运行测试
   - `auto_push`: 是否自动推送
6. 点击 **Run workflow**

### 工作流配置

```yaml
# .github/workflows/auto-archive.yml

name: Auto Archive

on:
  push:
    branches: [ main, master ]
  workflow_dispatch:
    inputs:
      message:
        description: 'Commit message'
        required: true
      run_tests:
        description: 'Run tests'
        required: true
        default: 'true'
```

---

## 🔐 安全设置

### GitHub Token

GitHub Actions 需要 `GITHUB_TOKEN` 权限：

```yaml
# 在 .github/workflows/auto-archive.yml 中
env:
  GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### 分支保护

建议设置分支保护规则：

1. 进入 GitHub 仓库 **Settings**
2. 选择 **Branches**
3. 点击 **Add branch protection rule**
4. 设置：
   - Branch name pattern: `main`
   - Require status checks to pass before merging
   - Require branches to be up to date before merging

---

## 💡 最佳实践

### 1. 提交信息规范

```bash
# 版本发布
-m "v0.4.0 完成"
-m "Release v0.4.0"

# 功能添加
-m "feat: 添加自动化进化系统"
-m "feat(auto): 自动存档功能"

# Bug 修复
-m "fix: 修复进化系统测试失败"
-m "fix(test): 修复测试超时问题"

# 文档更新
-m "docs: 更新存档指南"
-m "docs(readme): 更新使用说明"
```

### 2. 测试优先

```bash
# 推荐：始终运行测试后提交
python auto_archive.py -t -m "更新代码"

# 重要更新：测试后推送
python auto_archive.py -t -p -m "v0.4.0 完成"
```

### 3. 小步提交

```bash
# 好的做法
-m "feat: 添加测试基准"
-m "feat: 添加进化算法"
-m "feat: 添加自动系统"

# 不好的做法
-m "更新了很多代码"
```

### 4. 定期备份

```bash
# 创建远程备份
git push origin main:backup-main

# 创建标签
git tag -a v0.4.0 -m "Version 0.4.0"
git push origin v0.4.0
```

---

## 🔧 故障排除

### 问题 1: 测试失败

```bash
# 查看详细错误
python auto_archive.py -t -m "测试"

# 如果持续失败，跳过测试提交
python auto_archive.py -m "紧急修复" --no-test

# 然后修复问题后再正常提交
```

### 问题 2: 推送失败

```bash
# 检查远程仓库
git remote -v

# 添加远程仓库
git remote add origin https://github.com/username/twistor-lnn.git

# 重新推送
python auto_archive.py -p -m "重新推送"
```

### 问题 3: 需要回滚

```bash
# 自动回滚
python auto_archive.py --rollback

# 手动回滚
git reset --hard HEAD~1
```

### 问题 4: 备份残留

```bash
# 清理备份
rm -rf .git_backup

# 或使用脚本
python auto_archive.py --status  # 查看状态
```

---

## 📊 存档统计

### 查看存档历史

```bash
# 使用 git 命令
git log --oneline -10

# 使用存档日志
python -c "
import json
from datetime import datetime

with open('archive_log.json') as f:
    logs = json.load(f)

print(f\"总存档次数：{len(logs)}\")
print(f\"最近存档:\")
for log in logs[-5:]:
    print(f\"  {log['timestamp'][:10]}: {log['message']}\")
"
```

### 存档频率

| 类型 | 建议频率 |
|------|---------|
| **小更新** | 每天 |
| **功能完成** | 每功能 |
| **版本发布** | 每版本 |
| **紧急修复** | 立即 |

---

## 📋 总结

### 核心功能

```
┌─────────────────────────────────────────────────────────┐
│  GitHub 自动存档系统                                    │
├─────────────────────────────────────────────────────────┤
│  ✅ 自动测试验证                                        │
│  ✅ 自动提交到 Git                                      │
│  ✅ 测试通过则推送 (存档)                               │
│  ✅ 测试失败则回滚                                      │
│  ✅ GitHub Actions 集成                                 │
│  ✅ 完整的日志记录                                      │
└─────────────────────────────────────────────────────────┘
```

### 使用命令速查

```bash
# 基本提交
python auto_archive.py -m "消息"

# 测试后提交
python auto_archive.py -t -m "消息"

# 测试后推送
python auto_archive.py -t -p -m "消息"

# 查看状态
python auto_archive.py --status

# 回滚
python auto_archive.py --rollback
```

---

**版本**: 0.4.1  
**完成日期**: 2026-03-28  
**状态**: ✅ 自动存档系统完成

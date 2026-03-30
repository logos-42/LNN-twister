"""
Twistor-LNN GitHub 自动存档系统
================================
功能:
1. 自动测试验证
2. 自动提交到 Git
3. 测试通过则推送 (存档)
4. 测试失败则回滚

使用方式:
    python auto_archive.py --message "v0.4.0 完成"
    python auto_archive.py --test --message "v0.4.0 完成"
    python auto_archive.py --rollback
"""

import os
import sys
import subprocess
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List


class GitArchiveSystem:
    """Git 自动存档系统"""
    
    def __init__(self, repo_path: str = '.'):
        self.repo_path = Path(repo_path)
        self.backup_path = self.repo_path / '.git_backup'
        self.log_path = self.repo_path / 'archive_log.json'
        self.logs = self._load_logs()
    
    def _load_logs(self) -> List[dict]:
        """加载存档日志"""
        if self.log_path.exists():
            with open(self.log_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save_logs(self):
        """保存存档日志"""
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
    
    def _run_git(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """运行 git 命令"""
        result = subprocess.run(
            ['git'] + args,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=check
        )
        return result
    
    def check_git_status(self) -> Tuple[bool, str]:
        """检查 git 状态"""
        try:
            result = self._run_git(['status', '--porcelain'], check=False)
            has_changes = len(result.stdout.strip()) > 0
            
            result = self._run_git(['rev-parse', '--abbrev-ref', 'HEAD'], check=False)
            branch = result.stdout.strip()
            
            return has_changes, branch
        except Exception as e:
            return False, f"Error: {e}"
    
    def create_backup(self):
        """创建备份 (用于回滚)"""
        print("创建备份...")
        
        # 备份 .git 目录
        git_dir = self.repo_path / '.git'
        if git_dir.exists():
            if self.backup_path.exists():
                shutil.rmtree(self.backup_path)
            shutil.copytree(git_dir, self.backup_path)
            print(f"  ✓ 已备份 .git 目录到 {self.backup_path}")
        
        # 记录当前 commit
        try:
            result = self._run_git(['rev-parse', 'HEAD'])
            current_commit = result.stdout.strip()
            print(f"  ✓ 当前 commit: {current_commit[:8]}")
        except:
            current_commit = None
            print("  ⚠ 无法获取当前 commit")
        
        return current_commit
    
    def restore_backup(self):
        """恢复备份 (回滚)"""
        print("恢复备份 (回滚)...")
        
        if not self.backup_path.exists():
            print("  ✗ 备份不存在，无法回滚")
            return False
        
        # 恢复 .git 目录
        git_dir = self.repo_path / '.git'
        if git_dir.exists():
            shutil.rmtree(git_dir)
        shutil.copytree(self.backup_path, git_dir)
        print(f"  ✓ 已恢复 .git 目录")
        
        # 清理备份
        shutil.rmtree(self.backup_path)
        print(f"  ✓ 已清理备份")
        
        return True
    
    def cleanup_backup(self):
        """清理备份"""
        if self.backup_path.exists():
            shutil.rmtree(self.backup_path)
            print("已清理备份")
    
    def run_tests(self) -> Tuple[bool, str]:
        """运行测试"""
        print("运行测试...")
        
        test_results = []
        
        # 测试 1: 导入测试
        print("  测试 1: 导入测试...")
        try:
            import torch
            from twistor_100m_config import StackedTwistorLNN, TwistorLNNConfig
            
            config = TwistorLNNConfig.small()
            model = StackedTwistorLNN(config)
            
            x = torch.randn(10, 1, 1)
            with torch.no_grad():
                y = model(x)
            
            assert y.shape == (10, 1, 1), f"输出形状错误：{y.shape}"
            print("    ✓ 导入测试通过")
            test_results.append(("import_test", True))
        except Exception as e:
            print(f"    ✗ 导入测试失败：{e}")
            test_results.append(("import_test", False))
        
        # 测试 2: 进化系统测试
        print("  测试 2: 进化系统测试...")
        try:
            from auto_evolution import AutoEvolutionSystem, ArchitectureConfig
            
            config = ArchitectureConfig(hidden_dim=64, n_layers=2)
            assert config.hidden_dim == 64
            assert config.n_layers == 2
            print("    ✓ 进化系统测试通过")
            test_results.append(("evolution_test", True))
        except Exception as e:
            print(f"    ✗ 进化系统测试失败：{e}")
            test_results.append(("evolution_test", False))
        
        # 测试 3: 文件完整性测试
        print("  测试 3: 文件完整性测试...")
        try:
            required_files = [
                'twistor_lnn.py',
                'twistor_100m_config.py',
                'auto_evolution.py',
                'README.md',
                'requirements.txt',
            ]
            
            for file in required_files:
                assert (self.repo_path / file).exists(), f"缺少文件：{file}"
            
            print("    ✓ 文件完整性测试通过")
            test_results.append(("file_integrity_test", True))
        except Exception as e:
            print(f"    ✗ 文件完整性测试失败：{e}")
            test_results.append(("file_integrity_test", False))
        
        # 总结
        all_passed = all(result[1] for result in test_results)
        
        if all_passed:
            print("\n✓ 所有测试通过")
        else:
            failed = [name for name, passed in test_results if not passed]
            print(f"\n✗ 测试失败：{', '.join(failed)}")
        
        return all_passed, ", ".join([name for name, _ in test_results])
    
    def stage_changes(self):
        """暂存更改"""
        print("暂存更改...")
        
        # 添加所有更改
        self._run_git(['add', '-A'])
        
        # 检查暂存状态
        result = self._run_git(['diff', '--cached', '--name-only'])
        staged_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        print(f"  ✓ 已暂存 {len(staged_files)} 个文件")
        for f in staged_files[:10]:  # 只显示前 10 个
            print(f"    - {f}")
        if len(staged_files) > 10:
            print(f"    ... 还有 {len(staged_files) - 10} 个文件")
        
        return staged_files
    
    def commit(self, message: str) -> Optional[str]:
        """提交"""
        print(f"提交：{message}")
        
        try:
            # 设置用户信息 (如果未设置)
            self._run_git(['config', 'user.name', 'Twistor-LNN-Bot'], check=False)
            self._run_git(['config', 'user.email', 'bot@twistor-lnn.local'], check=False)
            
            # 提交
            self._run_git(['commit', '-m', message])
            
            # 获取 commit hash
            result = self._run_git(['rev-parse', 'HEAD'])
            commit_hash = result.stdout.strip()
            
            print(f"  ✓ 提交成功：{commit_hash[:8]}")
            
            return commit_hash
        except Exception as e:
            print(f"  ✗ 提交失败：{e}")
            return None
    
    def push(self, remote: str = 'origin', branch: Optional[str] = None) -> bool:
        """推送到远程"""
        print(f"推送到 {remote}...")
        
        try:
            # 获取当前分支
            if branch is None:
                result = self._run_git(['rev-parse', '--abbrev-ref', 'HEAD'])
                branch = result.stdout.strip()
            
            # 推送
            result = subprocess.run(
                ['git', 'push', remote, branch],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"  ✓ 推送成功到 {remote}/{branch}")
                return True
            else:
                print(f"  ✗ 推送失败：{result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("  ✗ 推送超时")
            return False
        except Exception as e:
            print(f"  ✗ 推送失败：{e}")
            return False
    
    def archive(self, message: str, run_test: bool = True, auto_push: bool = True) -> bool:
        """
        完整存档流程
        
        Args:
            message: 提交信息
            run_test: 是否运行测试
            auto_push: 测试通过后是否自动推送
        
        Returns:
            是否成功
        """
        print("=" * 70)
        print("Twistor-LNN 自动存档系统")
        print("=" * 70)
        print(f"提交信息：{message}")
        print(f"运行测试：{run_test}")
        print(f"自动推送：{auto_push}")
        print()
        
        # 1. 创建备份
        print("步骤 1: 创建备份")
        print("-" * 40)
        old_commit = self.create_backup()
        print()
        
        # 2. 运行测试
        if run_test:
            print("步骤 2: 运行测试")
            print("-" * 40)
            test_passed, test_info = self.run_tests()
            print()
            
            if not test_passed:
                print("✗ 测试失败，执行回滚")
                self.restore_backup()
                return False
        else:
            print("步骤 2: 跳过测试")
            print()
        
        # 3. 暂存更改
        print("步骤 3: 暂存更改")
        print("-" * 40)
        self.stage_changes()
        print()
        
        # 4. 提交
        print("步骤 4: 提交")
        print("-" * 40)
        commit_hash = self.commit(message)
        
        if not commit_hash:
            print("✗ 提交失败，执行回滚")
            self.restore_backup()
            return False
        print()
        
        # 5. 推送 (如果测试通过且自动推送)
        if auto_push:
            print("步骤 5: 推送到远程")
            print("-" * 40)
            push_success = self.push()
            
            if not push_success:
                print("✗ 推送失败，但保留本地提交")
                # 不回滚，因为本地提交是成功的
        else:
            print("步骤 5: 跳过推送 (本地提交)")
            print()
        
        # 6. 清理备份
        print("步骤 6: 清理备份")
        print("-" * 40)
        self.cleanup_backup()
        print()
        
        # 7. 记录日志
        print("步骤 7: 记录日志")
        print("-" * 40)
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'commit': commit_hash,
            'test_passed': run_test,
            'pushed': auto_push,
        }
        self.logs.append(log_entry)
        self._save_logs()
        print(f"  ✓ 已记录到 {self.log_path}")
        print()
        
        # 完成
        print("=" * 70)
        print("✓ 存档完成")
        print(f"  提交：{commit_hash[:8]}")
        print(f"  信息：{message}")
        if auto_push:
            print(f"  状态：已推送到远程")
        else:
            print(f"  状态：本地提交 (未推送)")
        print("=" * 70)
        
        return True
    
    def rollback(self) -> bool:
        """回滚到上一个版本"""
        print("=" * 70)
        print("Twistor-LNN 回滚系统")
        print("=" * 70)
        
        # 尝试恢复备份
        if self.backup_path.exists():
            print("发现未完成的备份，执行回滚...")
            success = self.restore_backup()
        else:
            # 尝试回滚到上一个 commit
            print("尝试回滚到上一个 commit...")
            try:
                self._run_git(['reset', '--hard', 'HEAD~1'])
                print("  ✓ 已回滚到上一个 commit")
                success = True
            except Exception as e:
                print(f"  ✗ 回滚失败：{e}")
                success = False
        
        print()
        print("=" * 70)
        if success:
            print("✓ 回滚完成")
        else:
            print("✗ 回滚失败")
        print("=" * 70)
        
        return success
    
    def show_status(self):
        """显示状态"""
        print("=" * 70)
        print("Twistor-LNN Git 状态")
        print("=" * 70)
        
        has_changes, branch = self.check_git_status()
        print(f"当前分支：{branch}")
        print(f"有未提交更改：{'是' if has_changes else '否'}")
        print()
        
        # 显示最近的存档记录
        print("最近的存档记录:")
        for log in self.logs[-5:]:
            print(f"  - {log['timestamp'][:10]}: {log['message']} ({log['commit'][:8]})")
        print()
        
        # 显示 git status
        print("Git 状态:")
        try:
            result = self._run_git(['status', '--short'])
            if result.stdout.strip():
                print(result.stdout)
            else:
                print("  工作区干净")
        except:
            print("  无法获取 git 状态")
        
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Twistor-LNN GitHub 自动存档系统')
    parser.add_argument('-m', '--message', type=str, help='提交信息')
    parser.add_argument('-t', '--test', action='store_true', help='运行测试')
    parser.add_argument('-p', '--push', action='store_true', help='自动推送到远程')
    parser.add_argument('--rollback', action='store_true', help='回滚到上一个版本')
    parser.add_argument('--status', action='store_true', help='显示状态')
    parser.add_argument('--no-test', action='store_true', help='跳过测试')
    parser.add_argument('--no-push', action='store_true', help='不推送 (仅本地提交)')
    
    args = parser.parse_args()
    
    archive_system = GitArchiveSystem()
    
    # 回滚
    if args.rollback:
        archive_system.rollback()
        return
    
    # 显示状态
    if args.status:
        archive_system.show_status()
        return
    
    # 存档
    if args.message:
        run_test = args.test and not args.no_test
        auto_push = args.push and not args.no_push
        
        # 默认行为：如果有 --test 标志则运行测试，否则不运行
        # 默认不自动推送，除非指定 --push
        success = archive_system.archive(
            message=args.message,
            run_test=run_test,
            auto_push=auto_push,
        )
        
        sys.exit(0 if success else 1)
    
    # 没有提供参数，显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()

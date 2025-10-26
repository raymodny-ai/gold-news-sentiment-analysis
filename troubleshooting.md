# 🔧 故障排除指南

## 仪表板访问问题

如果您无法访问仪表板 `http://localhost:8501`，请尝试以下解决方案：

### 1. 尝试不同的URL地址

仪表板可能在以下地址运行：

- ✅ `http://localhost:8501`
- ✅ `http://127.0.0.1:8501`
- ✅ `http://0.0.0.0:8501`

在浏览器中逐一尝试这些地址。

### 2. 检查仪表板是否正在运行

```bash
# 检查8501端口是否被占用
netstat -an | findstr :8501

# 如果没有运行，启动仪表板
python start_dashboard.py
```

### 3. 重新启动仪表板

```bash
# 停止当前仪表板进程（如果有的话）
# 然后重新启动
python start_dashboard.py
```

### 4. 检查防火墙设置

确保Windows防火墙允许8501端口的连接：

1. 打开Windows防火墙设置
2. 点击"高级设置"
3. 添加入站规则允许8501端口的TCP连接

### 5. 检查浏览器设置

1. 尝试不同的浏览器（Chrome、Firefox、Edge）
2. 清除浏览器缓存
3. 禁用浏览器扩展

### 6. 检查网络配置

```bash
# 检查IP配置
ipconfig

# 测试本地连接
ping localhost
ping 127.0.0.1
```

### 7. 启动诊断

运行系统诊断脚本：

```bash
# 检查系统状态
python system_status.py

# 测试仪表板
python test_dashboard.py
```

## 常见错误及解决方案

### "Connection Refused" 或 "Unable to connect"
- 仪表板没有运行 → 启动仪表板
- 端口被占用 → 停止占用该端口的程序
- 防火墙阻止 → 添加防火墙规则

### "Site cannot be reached" 或 "DNS_PROBE_FINISHED_NXDOMAIN"
- URL地址错误 → 使用正确的地址
- 网络配置问题 → 检查hosts文件

### "This page isn't working" 或 "ERR_CONNECTION_RESET"
- 服务器崩溃 → 重新启动仪表板
- 内存不足 → 关闭其他程序

## 快速修复步骤

1. **确认仪表板运行**:
   ```bash
   python start_dashboard.py
   ```

2. **测试连接**:
   ```bash
   curl http://localhost:8501
   ```

3. **访问正确的URL**:
   - 在浏览器中访问 `http://localhost:8501`
   - 如果不行，尝试 `http://127.0.0.1:8501`

4. **如果还是不行**:
   - 运行 `python test_dashboard.py`
   - 查看错误信息
   - 按照提示操作

## 高级故障排除

### 检查Streamlit日志
```bash
# 查看Streamlit运行日志
python -c "
import subprocess
proc = subprocess.Popen(['python', '-m', 'streamlit', 'run', 'dashboard/app.py'],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print('Streamlit output:', proc.communicate()[0].decode())
"
```

### 检查端口占用
```bash
# Windows
netstat -an | findstr :8501

# 如果被占用，找到进程
netstat -ano | findstr :8501
# 然后使用任务管理器结束该进程
```

### 重置网络
```bash
# 重置TCP/IP堆栈
netsh int ip reset
netsh winsock reset
```

## 联系支持

如果以上方法都无法解决问题：

1. 运行 `python system_status.py` 获取系统状态
2. 运行 `python test_dashboard.py` 获取详细错误信息
3. 截图错误信息并联系支持

---

**记住**: 仪表板在 `http://localhost:8501` 运行，但有时需要尝试不同的URL变体。

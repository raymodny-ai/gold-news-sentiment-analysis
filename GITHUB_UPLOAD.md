# 📤 如何上传项目到GitHub

## ✅ 已完成的准备工作

- ✅ Git仓库已初始化
- ✅ 所有文件已添加并提交
- ✅ .gitignore 文件已配置
- ✅ README.md 已创建
- ✅ LICENSE 文件已添加

## 🚀 上传步骤

### 步骤1: 在GitHub上创建新仓库

1. 访问 https://github.com/new
2. 填写仓库信息:
   - **Repository name**: `gold-news-sentiment-analysis` (或您喜欢的名称)
   - **Description**: `黄金新闻情感分析和价格预测系统 - AI-powered gold news sentiment analysis and price prediction system`
   - **Visibility**: 选择 Public (公开) 或 Private (私有)
   - ⚠️ **重要**: 不要勾选以下选项 (因为我们已经有这些文件):
     - ❌ Add a README file
     - ❌ Add .gitignore
     - ❌ Choose a license
3. 点击 **"Create repository"** 按钮

### 步骤2: 连接并推送到GitHub

创建仓库后,GitHub会显示一个页面。请按以下步骤操作:

#### 在PowerShell中运行:

```powershell
# 进入项目目录
cd "F:\Financial Project\gold news"

# 添加远程仓库 (替换 YOUR_USERNAME 和 REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 将分支重命名为main (GitHub的默认分支名)
git branch -M main

# 推送代码到GitHub
git push -u origin main
```

### 步骤3: 验证上传

上传完成后:
1. 刷新GitHub仓库页面
2. 您应该能看到所有文件
3. README.md 会自动显示在仓库首页

## 🔐 如果需要身份验证

### 使用Personal Access Token (推荐)

如果推送时要求输入密码,您需要使用Personal Access Token:

1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token" → "Generate new token (classic)"
3. 选择权限:
   - ✅ repo (所有仓库权限)
4. 点击 "Generate token"
5. **复制生成的token** (只显示一次!)
6. 推送时,用这个token替代密码

### 使用SSH (可选)

如果您配置了SSH密钥:

```powershell
# 使用SSH URL
git remote set-url origin git@github.com:YOUR_USERNAME/REPO_NAME.git
git push -u origin main
```

## 📝 后续更新

以后修改代码后,使用以下命令更新GitHub:

```powershell
cd "F:\Financial Project\gold news"

# 查看修改的文件
git status

# 添加所有修改
git add .

# 提交修改
git commit -m "描述您的修改"

# 推送到GitHub
git push
```

## 🎯 快速命令参考

```powershell
# 查看当前状态
git status

# 查看提交历史
git log --oneline

# 查看远程仓库
git remote -v

# 拉取最新代码
git pull

# 创建新分支
git checkout -b feature-name

# 切换分支
git checkout main
```

## ⚠️ 注意事项

1. **不要提交敏感信息**:
   - API密钥
   - 密码
   - 数据库文件
   - (这些已在.gitignore中排除)

2. **数据库文件**:
   - `gold_news.db` 已被.gitignore排除
   - 其他用户需要自己创建数据库

3. **ML模型文件**:
   - 保存的模型文件(.pkl, .h5)已被排除
   - 模型会在首次运行时自动下载

## 🌟 添加GitHub徽章

上传成功后,可以在README.md顶部添加徽章:

```markdown
[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/REPO_NAME?style=social)](https://github.com/YOUR_USERNAME/REPO_NAME/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/YOUR_USERNAME/REPO_NAME?style=social)](https://github.com/YOUR_USERNAME/REPO_NAME/network/members)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```

## 🤝 邀请协作者

如果想邀请其他人协作:
1. 进入仓库页面
2. Settings → Collaborators
3. 输入GitHub用户名邀请

---

需要帮助? 查看 [GitHub文档](https://docs.github.com/zh) 或提Issue!


@echo off
chcp 65001 >nul
echo ========================================
echo  上传项目到GitHub
echo ========================================
echo.

REM 检查是否已配置远程仓库
git remote -v | findstr "origin" >nul
if %errorlevel% neq 0 (
    echo [提示] 首次上传,需要配置GitHub仓库地址
    echo.
    echo 请按以下步骤操作:
    echo 1. 访问 https://github.com/new 创建新仓库
    echo 2. 复制仓库URL (例如: https://github.com/username/repo.git^)
    echo.
    set /p REPO_URL="请粘贴仓库URL: "
    
    echo.
    echo [执行] 添加远程仓库...
    git remote add origin %REPO_URL%
    
    echo [执行] 重命名分支为main...
    git branch -M main
    
    echo [执行] 首次推送...
    git push -u origin main
    
    if %errorlevel% equ 0 (
        echo.
        echo ✓ 上传成功!
        echo ✓ 您的项目已发布到GitHub
        echo.
        echo 访问您的仓库: %REPO_URL:~0,-4%
    ) else (
        echo.
        echo ✗ 上传失败!
        echo.
        echo 可能的原因:
        echo - 需要GitHub身份验证
        echo - 请使用Personal Access Token作为密码
        echo - 或配置SSH密钥
        echo.
        echo 获取Token: https://github.com/settings/tokens
    )
) else (
    echo [提示] 远程仓库已配置
    echo.
    
    REM 检查是否有未提交的更改
    git status --porcelain | findstr "^" >nul
    if %errorlevel% equ 0 (
        echo [检测] 发现未提交的更改
        echo.
        git status
        echo.
        set /p COMMIT_MSG="请输入提交说明: "
        
        echo.
        echo [执行] 添加文件...
        git add .
        
        echo [执行] 提交更改...
        git commit -m "%COMMIT_MSG%"
    ) else (
        echo [检测] 没有新的更改需要提交
    )
    
    echo.
    echo [执行] 推送到GitHub...
    git push
    
    if %errorlevel% equ 0 (
        echo.
        echo ✓ 更新成功!
    ) else (
        echo.
        echo ✗ 推送失败!
        echo 请检查网络连接和GitHub凭据
    )
)

echo.
echo ========================================
pause


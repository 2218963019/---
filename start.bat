@echo off
chcp 65001 >nul
echo ========================================
echo   AI模拟答辩智能体 - 启动中...
echo ========================================
echo.
echo 启动后请在浏览器打开: http://localhost:8000/app
echo 按 Ctrl+C 停止服务器
echo.
py -m uvicorn api_server:app --host 0.0.0.0 --port 8000
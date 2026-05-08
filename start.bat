@echo off

chcp 65001 >nul

title AI Proxy Manager



cd /d "%~dp0"



echo.

echo  +============================================+

echo  ^|        AI Proxy Service Manager            ^|

echo  +============================================+

echo  ^|  [1] DeepSeek Proxy  (Port 5000)           ^|

echo  ^|  [2] GLM-5 Proxy     (Port 5001)           ^|

echo  ^|  [3] Aliyun Bailian  (Port 5002)           ^|

echo  ^|  [Q] Exit                                  ^|

echo  +============================================+

echo.



set /p choice="Select [1/2/3/Q]: "



if /i "%choice%"=="Q" goto :end

if "%choice%"=="1" goto :deepseek

if "%choice%"=="2" goto :glm

if "%choice%"=="3" goto :aliyun

goto :end



:deepseek

echo.

echo [DeepSeek] Starting...

if exist venv\Scripts\activate.bat (

    call venv\Scripts\activate.bat

)

python deepseek_proxy.py

goto :end



:glm

echo.

echo [GLM-5] Starting...

if exist venv\Scripts\activate.bat (

    call venv\Scripts\activate.bat

)

python glm_proxy.py

goto :end



:aliyun

echo.

echo [Aliyun] Starting...

if exist venv\Scripts\activate.bat (

    call venv\Scripts\activate.bat

)

python aliyun_proxy.py

goto :end



:end
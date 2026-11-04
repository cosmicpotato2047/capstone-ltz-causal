# 학과 제출본 빌드.  PowerShell 에서:  .\paper\build_v2.ps1
# BibTeX 을 쓰지 않는다 (참고문헌은 generated/refs_v2.tex 로 미리 찍는다).
# 목차와 상호참조가 자리를 잡으려면 xelatex 두 번이면 된다.
$tex = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64"
if (-not (Test-Path "$tex\xelatex.exe")) { throw "MiKTeX not found: $tex" }
Push-Location $PSScriptRoot
& "$tex\xelatex.exe" -interaction=nonstopmode main_ko_v2.tex | Out-Null
& "$tex\xelatex.exe" -interaction=nonstopmode main_ko_v2.tex |
  Select-String -Pattern "^!|Output written|undefined|Warning: Reference"
Pop-Location
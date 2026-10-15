# 최종 보고서 빌드.  PowerShell 에서:  .\paper\build.ps1
# xelatex -> bibtex -> xelatex -> xelatex (목차와 참고문헌이 자리를 잡으려면 네 번)
$tex = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64"
if (-not (Test-Path "$tex\xelatex.exe")) { throw "MiKTeX not found: $tex" }
Push-Location $PSScriptRoot
& "$tex\xelatex.exe" -interaction=nonstopmode main_ko.tex | Out-Null
& "$tex\bibtex.exe"  main_ko | Out-Null
& "$tex\xelatex.exe" -interaction=nonstopmode main_ko.tex | Out-Null
& "$tex\xelatex.exe" -interaction=nonstopmode main_ko.tex |
  Select-String -Pattern "^!|Output written|undefined"
Pop-Location
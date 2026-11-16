# Launch Smart Jira MCP Server in Stdio Mode
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "Starting Smart Jira MCP Server on stdio transport..."
python (Join-Path $ScriptDir "server.py")

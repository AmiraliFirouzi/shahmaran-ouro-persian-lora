param(
  [Parameter(Mandatory)][string]$Code,
  [string]$Base = "",
  [string]$Token = "",
  [string]$Session = ""
)
# No secrets stored here. Pass -Base/-Token at call time, or set env vars.
if (-not $Base)   { $Base  = $env:MOLAB_URL }
if (-not $Token)  { $Token = $env:MOLAB_TOKEN }
if (-not $Base -or -not $Token) { Write-Error "missing -Base/-Token (or MOLAB_URL/MOLAB_TOKEN)"; exit 1 }
$auth = @()
if ($Token) { $auth = @("-H", "Authorization: Bearer $Token") }

if (-not $Session) {
  $s = & curl.exe -s @auth --connect-timeout 10 "$Base/api/sessions" 2>&1
  if ($s -match "terminated" -or $s -notmatch "^\{") { Write-Error "no sessions: $s"; exit 1 }
  $ids = ($s | ConvertFrom-Json).PSObject.Properties.Name
  if ($ids.Count -ne 1) { Write-Error "sessions: $($ids -join ', ')"; exit 1 }
  $Session = $ids[0]
}
Write-Verbose "session=$Session"

$tmp = Join-Path ([IO.Path]::GetTempPath()) "mc_body.json"
[IO.File]::WriteAllText($tmp, (@{ code = $Code } | ConvertTo-Json -Compress))
$raw = & curl.exe -sN -X POST "$Base/api/kernel/execute" `
  -H "Content-Type: application/json" `
  -H "Marimo-Session-Id: $Session" `
  @auth --data-binary "@$tmp" 2>&1

$out = New-Object System.Text.StringBuilder
$event = ""
$success = $true
foreach ($line in ($raw -split "`n")) {
  $line = $line.TrimEnd("`r")
  if ($line -like "event:*") { $event = $line.Substring(6).Trim() }
  elseif ($line -like "data:*") {
    $payload = $line.Substring(5).Trim()
    try { $j = $payload | ConvertFrom-Json } catch { continue }
    if ($event -eq "stdout") { [void]$out.Append($j.data) }
    elseif ($event -eq "stderr") { [void]$out.Append($j.data) }
    elseif ($event -eq "done") {
      if ($j.success -eq $false) { $success = $false }
      elseif ($j.output.data) { [void]$out.Append($j.output.data) }
    }
  }
}
if (-not $success) { Write-Error "EXEC FAILED" }
$out.ToString()

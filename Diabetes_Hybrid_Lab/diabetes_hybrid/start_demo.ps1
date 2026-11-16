$ErrorActionPreference = 'Stop'
$runtimeDir = Join-Path $env:LOCALAPPDATA 'DiabetesHybridRuntime'
$neo4jDir = Join-Path $runtimeDir 'neo4j-community-5.26.33'
$neo4jPort = 7689
$desktopSettings = Join-Path $runtimeDir 'neo4j-desktop.json'
if (-not $env:NEO4J_URI -and -not (Test-Path -LiteralPath $desktopSettings -PathType Leaf)) {
    $listener = Get-NetTCPConnection -LocalPort $neo4jPort -State Listen -ErrorAction SilentlyContinue
    if (-not $listener) {
        $jdkDir = Get-ChildItem $runtimeDir -Directory -Filter 'jdk-*' -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending | Select-Object -First 1
        if (-not $jdkDir -or -not (Test-Path (Join-Path $neo4jDir 'bin/neo4j.ps1'))) {
            throw 'Neo4j portable chưa có. Hãy xem mục cài Neo4j trong README.md.'
        }
        $env:JAVA_HOME = $jdkDir.FullName
        $env:NEO4J_HOME = $neo4jDir
        $script = Join-Path $neo4jDir 'bin/neo4j.ps1'
        Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden -ArgumentList @(
            '-NoProfile', '-File', ('"' + $script + '"'), 'console'
        ) -WorkingDirectory $neo4jDir -RedirectStandardOutput (Join-Path $runtimeDir 'neo4j-console.log') `
          -RedirectStandardError (Join-Path $runtimeDir 'neo4j-error.log') | Out-Null
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            Start-Sleep -Seconds 1
            $listener = Get-NetTCPConnection -LocalPort $neo4jPort -State Listen -ErrorAction SilentlyContinue
            if ($listener) { break }
        }
        if (-not $listener) { throw 'Neo4j chưa mở cổng 7689. Xem log trong DiabetesHybridRuntime.' }
    }
}
$python = (Get-Command python -ErrorAction Stop).Source
& $python (Join-Path $PSScriptRoot 'app.py')
exit $LASTEXITCODE


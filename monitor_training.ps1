# Live training progress monitor
$log = "training_run3.log"
$lastPos = 0

Write-Host "=== XTTS Training Monitor ===" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop monitoring`n" -ForegroundColor Gray

while ($true) {
    if (Test-Path $log) {
        $content = Get-Content $log -Encoding utf8 -Raw

        # Check for current epoch/step
        $epochMatch = $content | Select-String "EPOCH: (\d+)/(\d+)" | Select-Object -Last 1
        $stepMatch = $content | Select-String "STEP: (\d+)/(\d+)" | Select-Object -Last 1
        $lossMatch = $content | Select-String "loss: [\d.]+" | Select-Object -Last 1

        if ($epochMatch -and $stepMatch) {
            $e = $epochMatch.Matches[0].Groups[1].Value
            $emax = $epochMatch.Matches[0].Groups[2].Value
            $s = $stepMatch.Matches[0].Groups[1].Value
            $smax = $stepMatch.Matches[0].Groups[2].Value
            $loss = if ($lossMatch) { ($lossMatch.Line -replace '.*loss: ([\d.]+).*', '$1') } else { "?" }

            $pct = [int](($e * 189 + $s) * 100 / (20 * 189))
            $bar = "=" * ($pct / 2) + " " * (50 - $pct/2)

            Write-Host "`r[$bar] $pct% | EPOCH: $e/$emax | STEP: $s/$smax | loss: $loss" -NoNewline -ForegroundColor Green
        }

        # Check for eval
        if ($content -match "EVALUATION" -and $content -notmatch "eval_loss.*eval_loss") {
            Write-Host "`n[EVAL PASS RUNNING...]" -ForegroundColor Yellow
        }
    }

    Start-Sleep -Seconds 5
}

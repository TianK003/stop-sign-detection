<#
read_inference_time_swd.ps1 -- fully off-cloud on-device inference latency.

Reads the live inference timer of the running object-detection demo on the
STM32H747I-DISCO over ST-LINK (SWD hotplug, no halt, no reflash). The demo
brackets ai_run() with HAL_GetTick() timestamps stored in the global
App_Config (AppConfig_TypeDef); this script reads Tinf_start/Tinf_stop and
reports the difference in ms. This is the off-cloud counterpart to the ST Edge
AI Developer Cloud "Duration (ms)" figure -- no cloud, no firmware change.

Measured 2026-05-30 on the 256k2 champion: 329 ms (vs Dev Cloud 320.9 ms; the
~8 ms gap is 1 ms tick quantization + real camera/LTDC/SDRAM bus contention
that the isolated validation FW does not see).

ADDRESS IS BUILD-SPECIFIC. After any rebuild/reflash, re-derive the App_Config
address from the .map file:
    Select-String -Path <..._CM7.map> -Pattern "App_Config"
then add the AppConfig_TypeDef offset of Tinf_start. For the current build the
struct front is: nn_inference_time(0) name(4) proba(8) new_frame_ready(12,pad)
mirror_flip(16) cropping(20) red_blue_swap(24) PixelFormatConv(28) lcd_sync(32)
Tinf_start(36=0x24) Tinf_stop(40) Tfps_start(44) Tfps_stop(48). So
TinfAddr = App_Config + 0x24  (currently 0xD0A0D000 + 0x24 = 0xD0A0D024).

Usage:
    pwsh scripts/read_inference_time_swd.ps1
    pwsh scripts/read_inference_time_swd.ps1 -TinfAddr 0xD0A0D024 -Samples 30
#>
param(
    [string]$Cli = "C:/Program Files/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI.exe",
    [string]$TinfAddr = "0xD0A0D024",   # &App_Config + offsetof(Tinf_start)
    [int]$Samples = 30
)

$rows = @(); $seen = @{}
$addrTag = $TinfAddr.ToUpper() -replace '^0X','0x'
for ($i = 0; $i -lt $Samples; $i++) {
    # NOTE: -r32 count is in BYTES; 0x10 = 4 words (Tinf_start, Tinf_stop, Tfps_start, Tfps_stop)
    $out = & $Cli -c port=SWD mode=Hotplug ap=0 -r32 $TinfAddr 0x10 2>&1
    $m = [regex]::Match(($out -join "`n"),
        "$addrTag : ([0-9A-Fa-f]{8}) ([0-9A-Fa-f]{8}) ([0-9A-Fa-f]{8}) ([0-9A-Fa-f]{8})")
    if ($m.Success) {
        $ts = [Convert]::ToInt64($m.Groups[1].Value, 16)
        $te = [Convert]::ToInt64($m.Groups[2].Value, 16)
        $fs = [Convert]::ToInt64($m.Groups[3].Value, 16)
        $fe = [Convert]::ToInt64($m.Groups[4].Value, 16)
        $inf = $te - $ts; $loop = $fe - $fs
        if (-not $seen.ContainsKey($ts) -and $inf -gt 0 -and $inf -lt 5000) {
            $seen[$ts] = $true
            $rows += [pscustomobject]@{ inf = $inf; loop = $loop }
        }
    }
}

$inf = @($rows.inf)
if ($inf.Count -eq 0) {
    Write-Error "No samples parsed. Check the board is connected, the demo is running, and -TinfAddr matches the current .map (see header)."
    exit 1
}
$st = $inf | Measure-Object -Average -Minimum -Maximum
$sd = [math]::Sqrt(($inf | ForEach-Object { [math]::Pow($_ - $st.Average, 2) } | Measure-Object -Average).Average)
$lp = (@($rows.loop) | Measure-Object -Average).Average

Write-Host ""
Write-Host "==== On-device latency (off-cloud, STM32H747I-DISCO @400 MHz, live demo) ===="
Write-Host ("  unique frames    : {0}" -f $inf.Count)
Write-Host ("  INFERENCE (ai_run): mean={0:N1} ms  min={1}  max={2}  stdev={3:N1}" -f $st.Average, $st.Minimum, $st.Maximum, $sd)
Write-Host ("  full pipeline    : mean={0:N1} ms  => {1:N2} FPS" -f $lp, (1000.0 / $lp))
Write-Host ("  inference-only   : {0:N2} FPS" -f (1000.0 / $st.Average))
Write-Host "  (1 ms HAL_GetTick resolution; compare vs Dev Cloud cycle-accurate Duration)"
Write-Host "============================================================================="

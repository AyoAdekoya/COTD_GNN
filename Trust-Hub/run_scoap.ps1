# run_scoap.ps1

# Load SendKeys support
Add-Type -AssemblyName System.Windows.Forms

# Helper to log messages
$logFile = "$PSScriptRoot\scoap_run_log.txt"
function Log-Msg {
    param($Text)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts - $Text" | Out-File -FilePath $logFile -Append
}

# === Configuration ===
# $designs = @(
#     "c2670_T","c3540_T","c5315_T","c6288_T",
#     "s1423_T","s13207_T","s15850_T","s35932_T"
# )

$designs = @(
    "s1423_T","s13207_T","s15850_T","s35932_T"
)

$scoapExe  = "C:\Users\adeyo\OneDrive\Desktop\IntelResearch\SCOAP_Analysis_Tool-master\SCOAP_Analysis_Tool-master\SCOAPTOOL.EXE"
$inputBase = "C:\Users\adeyo\OneDrive\Desktop\IntelResearch\ICsDesign\Trojan_GNN\Trust-Hub\Scoap_Inputs\scoap_input"
$outputBase= "C:\Users\adeyo\OneDrive\Desktop\IntelResearch\ICsDesign\Trojan_GNN\Trust-Hub\Scoap_Outputs\SCOAP_Output"

# Create Shell COM for AppActivate if needed
$wshell = New-Object -ComObject WScript.Shell

# Launch SCOAP tool once
Start-Process -FilePath $scoapExe
Start-Sleep -Seconds 7
$wshell.AppActivate("Testability Measurement Tool") | Out-Null
Start-Sleep -Seconds 1
Log-Msg "Launched and focused SCOAP tool"

# Outer loop over design prefixes
foreach ($prefix in $designs) {
    for ($num = 1; $num -le 219; $num++) {
        $suffix = $num.ToString("000")
        $inputFile  = "$inputBase$prefix$suffix.txt"
        $outputFile = "$outputBase$prefix$suffix.txt"

        if (Test-Path $inputFile) {
            Log-Msg "Processing file: $inputFile"

            # Open input (Ctrl+O)
            Start-Sleep -Milliseconds 500
            $wshell.AppActivate("Testability Measurement Tool") | Out-Null
            # Start-Sleep -Milliseconds 200
            [System.Windows.Forms.SendKeys]::SendWait("^o")
            Start-Sleep -Milliseconds 500

            # [System.Windows.Forms.SendKeys]::SendWait("$inputFile")
            [System.Windows.Forms.Clipboard]::SetText($inputFile)   
            [System.Windows.Forms.SendKeys]::SendWait('^v')
            Start-Sleep -Milliseconds 100
            # [System.Windows.Forms.SendKeys]::SendWait('{ENTER}')
            # Start-Sleep -Milliseconds 250
            [System.Windows.Forms.SendKeys]::SendWait("{ENTER}")
            Log-Msg "Opened input: $inputFile"
            Start-Sleep -Seconds 3

            # Save output (Shift+Ctrl+S)
            [System.Windows.Forms.SendKeys]::SendWait("+^s")
            Start-Sleep -Milliseconds 500

            # [System.Windows.Forms.SendKeys]::SendWait("$outputFile")
            [System.Windows.Forms.Clipboard]::SetText($outputFile)   
            [System.Windows.Forms.SendKeys]::SendWait('^v')
            Start-Sleep -Milliseconds 100
            # Start-Sleep -Milliseconds 250
            [System.Windows.Forms.SendKeys]::SendWait("{ENTER}")
            Start-Sleep -Seconds 2
            Log-Msg "Saved output: $outputFile"
            Start-Sleep -Milliseconds 500
        }
        else {
            Log-Msg "Skipped missing file: $inputFile"
        }
    }
}

# Close tool (Alt+F4)
Start-Sleep -Milliseconds 500
[System.Windows.Forms.SendKeys]::SendWait("%{F4}")
Start-Sleep -Milliseconds 500
Log-Msg "Closed tool after processing"

Log-Msg "Batch processing complete."
[System.Windows.Forms.MessageBox]::Show("Batch complete.`nSee scoap_run_log.txt for details.","Done") | Out-Null
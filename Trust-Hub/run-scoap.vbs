Set WshShell = WScript.CreateObject("WScript.Shell")

logFilePath = "scoap_run_log.txt"
Set fso = CreateObject("Scripting.FileSystemObject")
Set logFile = fso.OpenTextFile(logFilePath, 2, True)

Sub logMsg(msg)
    logFile.WriteLine Now & " - " & msg
End Sub

On Error Resume Next

' Launch tool ONCE
WshShell.Run "scoaptool.exe"
WScript.Sleep 5000

WshShell.AppActivate "Testability Measurement Tool"
WScript.Sleep 5000
logMsg "Launched and focused SCOAP tool"

' === Design prefixes (edit as needed) ===
designs = Array("c2670_T", "c3540_T", "c5315_T", "c6288_T", "s1423_T", 
"s13207_T", "s15850_T", "s35932_T")

' === Outer loop: prefixes ===
For d = 0 To UBound(designs)
    prefix = designs(d)

    ' === Inner loop: numbers 001 to 219 ===
    For num = 1 To 219
        suffix = Right("000" & num, 3)  ' Pad to 3 digits

        ' Build file names
        inputFile = "Trojan_GNN\Trust-Hub\Scoap_Inputs\scoap_input" & prefix & suffix & ".txt"
        outputFile = "Trojan_GNN\Trust-Hub\Scoap_Outputs\SCOAP_Output" & prefix & suffix & ".txt"

        ' === Check if input file exists ===
        If fso.FileExists(inputFile) Then
            logMsg "Processing file: " & inputFile

            ' Open input
            WshShell.SendKeys "^o"
            WScript.Sleep 500

            WshShell.SendKeys inputFile
            WshShell.SendKeys "{ENTER}"
            logMsg "Opened input: " & inputFile
            WScript.Sleep 7000

            ' Save output
            WshShell.SendKeys "+^s"
            WScript.Sleep 500

            WshShell.SendKeys outputFile
            WshShell.SendKeys "{ENTER}"
            WScript.Sleep 2000
            logMsg "Saved output: " & outputFile
        Else
            logMsg "Skipped missing file: " & inputFile
        End If
    Next
Next

logMsg "All files processed."

' Optional: close tool
WshShell.SendKeys "%{F4}"
logMsg "Closed tool after processing."
logFile.Close

MsgBox "Batch processing complete. See scoap_run_log.txt for details."

On Error GoTo 0

Set WshShell = WScript.CreateObject("WScript.Shell")

logFilePath = "scoap_run_log.txt"
Set fso = CreateObject("Scripting.FileSystemObject")
Set logFile = fso.OpenTextFile(logFilePath, 2, True)

Sub logMsg(msg)
    logFile.WriteLine Now & " - " & msg
End Sub

On Error Resume Next

' Launch tool ONCE
WshShell.Run """C:\Users\adeyo\OneDrive\Desktop\IntelResearch\SCOAP_Analysis_Tool-master\SCOAP_Analysis_Tool-master\SCOAPTOOL.EXE""", 1, False
WScript.Sleep 5000

WshShell.AppActivate "Testability Measurement Tool"
WScript.Sleep 1000
logMsg "Launched and focused SCOAP tool"

' === Design prefixes (edit as needed) ===
designs = Array("c2670_T", "c3540_T", "c5315_T", "c6288_T", "s1423_T", "s13207_T", "s15850_T", "s35932_T")

' === Outer loop: prefixes ===
For d = 0 To UBound(designs)
    prefix = designs(d)

    ' === Inner loop: numbers 001 to 219 ===
    For num = 5 To 219
        suffix = Right("000" & num, 3)  ' Pad to 3 digits

        ' Build file names
        ' inputFile = "Trojan_GNN\Trust-Hub\Scoap_Inputs\scoap_input" & prefix & suffix & ".txt"
        ' outputFile = "Trojan_GNN\Trust-Hub\Scoap_Outputs\SCOAP_Output" & prefix & suffix & ".txt"

        inputFile = "C:\Users\adeyo\OneDrive\Desktop\IntelResearch\ICsDesign\Trojan_GNN\Trust-Hub\Scoap_Inputs\scoap_input" & prefix & suffix & ".txt"
        outputFile = "C:\Users\adeyo\OneDrive\Desktop\IntelResearch\ICsDesign\Trojan_GNN\Trust-Hub\Scoap_Outputs\SCOAP_Output" & prefix & suffix & ".txt"

        ' === Check if input file exists ===
        If fso.FileExists(inputFile) Then
            WScript.Sleep 500
            logMsg "Processing file: " & inputFile

            ' Open input
            WScript.Sleep 1500
            WshShell.AppActivate "Testability Measurement Tool"
            WScript.Sleep 200
            WshShell.SendKeys "^{o}", True
            WScript.Sleep 500

            WshShell.SendKeys inputFile
            WScript.Sleep 250
            WshShell.SendKeys "{ENTER}", True
            logMsg "Opened input: " & inputFile
            WScript.Sleep 3500

            ' Save output
            WshShell.SendKeys "+^s"
            WScript.Sleep 500

            WshShell.SendKeys outputFile
            WScript.Sleep 250
            WshShell.SendKeys "{ENTER}", True
            WScript.Sleep 2000
            logMsg "Saved output: " & outputFile
            WScript.Sleep 500
        Else
            WScript.Sleep 500
            logMsg "Skipped missing file: " & inputFile
            WScript.Sleep 500
        End If
    Next
Next

logMsg "All files processed."

' Optional: close tool
WScript.Sleep 500
WshShell.SendKeys "%{F4}"
WScript.Sleep 500
logMsg "Closed tool after processing."
logFile.Close

MsgBox "Batch processing complete. See scoap_run_log.txt for details."

On Error GoTo 0

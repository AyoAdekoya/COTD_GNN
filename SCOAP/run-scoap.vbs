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

For i = 0 To 29
    logMsg "Processing file #" & i

    ' Open input
    WshShell.SendKeys "^o"
    WScript.Sleep 500

    inputFile = "scoap_inputs\\scoap_input" & i & ".txt"
    WshShell.SendKeys inputFile
    WshShell.SendKeys "{ENTER}"
    logMsg "Opened input: " & inputFile
    WScript.Sleep 7000

    ' Save output
    WshShell.SendKeys "+^s"
    WScript.Sleep 500

    outputFile = "scoap_outputs\SCOAP_Output" & i & ".txt"
    WshShell.SendKeys outputFile
    WshShell.SendKeys "{ENTER}"
    WScript.Sleep 2000
    logMsg "Saved output: " & outputFile
Next

logMsg "All files processed."

' Optional: Close tool once at the end
WshShell.SendKeys "%{F4}"
logMsg "Closed tool after processing."
logFile.Close

MsgBox "Batch processing complete. See scoap_run_log.txt for details."

On Error GoTo 0
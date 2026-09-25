$ErrorActionPreference = 'Stop'
$resumeWord = New-Object -ComObject Word.Application
$resumeWord.Visible = $false
$resumeWord.DisplayAlerts = 0
try {
  foreach ($resumeLang in @('EN','CN')) {
    $resumeDocPath = Join-Path $PSScriptRoot ('Chihan_Gao_Quant_Resume_' + $resumeLang + '.docx')
    $resumePdfPath = Join-Path $PSScriptRoot ('Chihan_Gao_Quant_Resume_' + $resumeLang + '.pdf')
    $resumeDocument = $resumeWord.Documents.Open($resumeDocPath, $false, $true)
    $resumeDocument.ExportAsFixedFormat($resumePdfPath, 17)
    Write-Output ($resumeLang + ' pages: ' + $resumeDocument.ComputeStatistics(2))
    $resumeDocument.Close(0)
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($resumeDocument)
  }
} finally {
  $resumeWord.Quit()
  [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($resumeWord)
}

# Authenticode-sign files with the SSL.com OV certificate through eSigner CKA,
# the same flow as ThisIsMyPC's release signing:
#   1. CodeSignTool scan_code submits each file for the account's malware
#      blocker (needs the eSigner username, password and credential ID).
#   2. signtool signs with the CKA-backed certificate from Cert:\CurrentUser\My
#      and an RFC 3161 timestamp from SSL.com, retrying while the scan
#      approval propagates.
#   3. The signature and timestamp are verified.
# Each signed file costs one SSL.com signing credit.
#
#   .\installer\sign.ps1 -File dist\Mandelbloom-Setup-0.1.0.exe
#
# Defaults: the one "No More Secrets, LLC" code-signing certificate in the
# user store, ESIGNER_USERNAME and ESIGNER_CREDENTIAL_ID from the user
# environment, the password from ThisIsMyPC's DPAPI-saved copy if present,
# otherwise a private prompt. CodeSignTool is downloaded once into
# build\tools (hash-pinned) or reused from ThisIsMyPC's cache.
param(
    [Parameter(Mandatory)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string[]]$File,
    [string]$Thumbprint,
    [string]$CredentialId,
    [string]$Username,
    [Security.SecureString]$Password,
    [string]$CodeSignToolArchive,
    [string]$Description = 'Mandelbloom',
    [string]$TimestampUrl,
    [switch]$SkipScan
)

$ErrorActionPreference = 'Stop'
Import-Module Microsoft.PowerShell.Security -ErrorAction Stop
$root = Split-Path $PSScriptRoot -Parent
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'esigner-codesigntool.json') -Raw | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($TimestampUrl)) { $TimestampUrl = $manifest.timestampUrl }
$paths = @($File | ForEach-Object { (Resolve-Path -LiteralPath $_).Path })

# ---- certificate ----
$store = @(Get-ChildItem Cert:\CurrentUser\My)
if ([string]::IsNullOrWhiteSpace($Thumbprint)) {
    $candidates = @($store | Where-Object {
        $_.GetNameInfo([Security.Cryptography.X509Certificates.X509NameType]::SimpleName, $false) -eq $manifest.signer -and
        $_.HasPrivateKey -and (Get-Date) -ge $_.NotBefore -and (Get-Date) -le $_.NotAfter
    })
    if ($candidates.Count -ne 1) { throw "Expected one valid '$($manifest.signer)' certificate with a private key in Cert:\CurrentUser\My, found $($candidates.Count). Is eSigner CKA installed and logged in?" }
    $Thumbprint = $candidates[0].Thumbprint
}
$Thumbprint = $Thumbprint.ToUpperInvariant()
if ($Thumbprint -notmatch '^[0-9A-F]{40}$') { throw 'Thumbprint must be 40 hex characters.' }
$certificate = @($store | Where-Object { $_.Thumbprint -eq $Thumbprint })
if ($certificate.Count -ne 1) { throw "Certificate $Thumbprint not found in Cert:\CurrentUser\My." }
$certificate = $certificate[0]
if (-not $certificate.HasPrivateKey) { throw 'Certificate found but its private key (eSigner CKA) is not reachable.' }
$signerName = $certificate.GetNameInfo([Security.Cryptography.X509Certificates.X509NameType]::SimpleName, $false)
if ($signerName -ne $manifest.signer) { throw "Refusing unexpected signing identity: $signerName" }
$eku = $certificate.Extensions | Where-Object { $_.Oid.Value -eq '2.5.29.37' } | Select-Object -First 1
if (-not $eku -or ([Security.Cryptography.X509Certificates.X509EnhancedKeyUsageExtension]$eku).EnhancedKeyUsages.Value -notcontains '1.3.6.1.5.5.7.3.3') {
    throw 'Selected certificate is not a code-signing certificate.'
}

# ---- signtool ----
$signTool = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe" -ErrorAction SilentlyContinue |
    Sort-Object FullName -Descending | Select-Object -First 1
if (-not $signTool) { throw 'signtool.exe not found (Windows 10/11 SDK).' }

foreach ($p in $paths) {
    if ((Get-AuthenticodeSignature -LiteralPath $p).Status -ne 'NotSigned') { throw "Already signed: $p" }
}

# ---- malware scan (eSigner account blocker) ----
if (-not $SkipScan) {
    if ([string]::IsNullOrWhiteSpace($CredentialId)) { $CredentialId = $env:ESIGNER_CREDENTIAL_ID }
    if ([string]::IsNullOrWhiteSpace($CredentialId)) { $CredentialId = [Environment]::GetEnvironmentVariable('ESIGNER_CREDENTIAL_ID', 'User') }
    if ($CredentialId -notmatch '^[0-9a-fA-F-]{36}$') { throw 'eSigner credential ID (36 characters, beside the code-signing certificate in the SSL.com account) is required: -CredentialId or ESIGNER_CREDENTIAL_ID.' }
    if ([string]::IsNullOrWhiteSpace($Username)) { $Username = $env:ESIGNER_USERNAME }
    if ([string]::IsNullOrWhiteSpace($Username)) { $Username = [Environment]::GetEnvironmentVariable('ESIGNER_USERNAME', 'User') }
    if ([string]::IsNullOrWhiteSpace($Username)) { $Username = Read-Host 'SSL.com eSigner username' }
    if ([string]::IsNullOrWhiteSpace($Username)) { throw 'eSigner username is required.' }

    if (-not $Password) {
        $saved = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'ThisIsMyPC\ReleaseSigning\esigner-password.clixml'
        if (Test-Path -LiteralPath $saved -PathType Leaf) {
            $Password = Import-Clixml -LiteralPath $saved
            Write-Host 'Using the eSigner password saved by ThisIsMyPC (DPAPI).'
        }
    }
    if (-not $Password) { $Password = Read-Host 'SSL.com eSigner account password' -AsSecureString }
    if ($Password.Length -eq 0) { throw 'eSigner password is required.' }

    # CodeSignTool archive: ThisIsMyPC's verified cache, our cache, or a pinned download.
    $archiveName = "CodeSignTool-v$($manifest.codeSignTool.version)-windows.zip"
    if ([string]::IsNullOrWhiteSpace($CodeSignToolArchive)) {
        foreach ($c in @("$root\build\tools\$archiveName", "C:\Users\$env:USERNAME\Dev-Projects\thisismypc\artifacts\tool-cache\esigner\$archiveName")) {
            if (Test-Path -LiteralPath $c -PathType Leaf) { $CodeSignToolArchive = $c; break }
        }
    }
    if ([string]::IsNullOrWhiteSpace($CodeSignToolArchive)) {
        New-Item -ItemType Directory -Force "$root\build\tools" | Out-Null
        $CodeSignToolArchive = "$root\build\tools\$archiveName"
        Write-Host "Downloading pinned CodeSignTool $($manifest.codeSignTool.version) from SSL.com..."
        Invoke-WebRequest -UseBasicParsing -Uri $manifest.codeSignTool.archiveUrl -OutFile "$CodeSignToolArchive.download"
        Move-Item -LiteralPath "$CodeSignToolArchive.download" -Destination $CodeSignToolArchive -Force
    }
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $CodeSignToolArchive).Hash
    if ($hash -ne $manifest.codeSignTool.archiveSha256) { throw "CodeSignTool archive hash is $hash, expected $($manifest.codeSignTool.archiveSha256): $CodeSignToolArchive" }

    $tmp = Join-Path ([IO.Path]::GetTempPath()) ('mandelbloom-codesigntool-' + [guid]::NewGuid().ToString('N'))
    $bstr = [IntPtr]::Zero
    $plain = $null
    try {
        New-Item -ItemType Directory -Path $tmp | Out-Null
        Expand-Archive -LiteralPath $CodeSignToolArchive -DestinationPath $tmp
        $java = Join-Path $tmp $manifest.codeSignTool.javaRelativePath
        $jar = Join-Path $tmp $manifest.codeSignTool.jarRelativePath
        foreach ($t in @(@{ Path = $java; Hash = $manifest.codeSignTool.javaSha256 }, @{ Path = $jar; Hash = $manifest.codeSignTool.jarSha256 })) {
            if (-not (Test-Path -LiteralPath $t.Path -PathType Leaf)) { throw "CodeSignTool archive is missing $($t.Path)." }
            $h = (Get-FileHash -Algorithm SHA256 -LiteralPath $t.Path).Hash
            if ($h -ne $t.Hash) { throw "Extracted CodeSignTool file differs from the pin: $($t.Path)" }
        }
        $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Password)
        $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
        Push-Location $tmp
        try {
            foreach ($p in $paths) {
                Write-Host "Scanning for eSigner approval: $(Split-Path $p -Leaf)"
                # CodeSignTool has no protected password channel: the Java process
                # receives it on its command line (as in ThisIsMyPC).
                & $java -jar $jar scan_code "-credential_id=$CredentialId" "-input_file_path=$p" "-password=$plain" "-program_name=$Description" "-username=$Username"
                if ($LASTEXITCODE -ne 0) { throw "CodeSignTool scan failed ($LASTEXITCODE) for $p" }
            }
        }
        finally { Pop-Location }
    }
    finally {
        $plain = $null
        if ($bstr -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
        if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force }
    }
}

# ---- sign, retrying while the scan approval propagates to CKA ----
$backup = Join-Path ([IO.Path]::GetTempPath()) ('mandelbloom-sign-backup-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $backup | Out-Null
try {
    $i = 0
    $backups = @{}
    foreach ($p in $paths) { $i++; $b = Join-Path $backup "$i-$(Split-Path $p -Leaf)"; Copy-Item -LiteralPath $p -Destination $b; $backups[$p] = $b }
    $delays = @(5, 15, 30)
    $attempt = 0
    while ($true) {
        $attempt++
        $saved = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            $out = @(& $signTool.FullName sign /fd sha256 /tr $TimestampUrl /td sha256 /d $Description /sha1 $Thumbprint @paths 2>&1)
            $code = $LASTEXITCODE
        }
        finally { $ErrorActionPreference = $saved }
        $out | ForEach-Object { Write-Host $_ }
        if ($code -eq 0) { break }
        foreach ($p in $paths) { Copy-Item -LiteralPath $backups[$p] -Destination $p -Force }
        $pending = ($out -join "`n") -match 'needs to be scanned first'
        if (-not $pending -or $attempt -gt $delays.Count) { throw 'signtool failed.' }
        $d = $delays[$attempt - 1]
        Write-Host "eSigner approval not visible to CKA yet; retrying in $d s."
        Start-Sleep -Seconds $d
    }
}
finally {
    if (Test-Path -LiteralPath $backup) { Remove-Item -LiteralPath $backup -Recurse -Force }
}

foreach ($p in $paths) {
    & $signTool.FullName verify /pa /all $p
    if ($LASTEXITCODE -ne 0) { throw "signtool verification failed: $p" }
    $sig = Get-AuthenticodeSignature -LiteralPath $p
    if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Thumbprint -ne $Thumbprint -or $null -eq $sig.TimeStamperCertificate) {
        throw "Signature or timestamp validation failed: $p"
    }
    Write-Host "Signed and timestamped: $p"
}

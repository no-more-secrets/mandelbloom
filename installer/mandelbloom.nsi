; Mandelbloom installer (NSIS 3, Modern UI 2).
; Built by installer\build-installer.ps1, which passes VERSION, STAGE and
; OUTFILE. TESTMODE builds a per-user installer into %LOCALAPPDATA% for
; automated checks without a UAC prompt.

!ifndef VERSION
  !define VERSION "0.0.0"
!endif
!ifndef STAGE
  !define STAGE "..\build\installer\stage"
!endif
!ifndef OUTFILE
  !define OUTFILE "..\dist\Mandelbloom-Setup-${VERSION}.exe"
!endif
!define APPNAME "Mandelbloom"
!define COMPANY "NMS"
!define EXE "Mandelbloom.exe"
!define UNINSTKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"

Unicode true
SetCompressor /SOLID lzma
Name "${APPNAME}"
OutFile "${OUTFILE}"
BrandingText "${COMPANY}"

!ifdef TESTMODE
  RequestExecutionLevel user
  !define REGROOT HKCU
  !define PARENTDIR "$LOCALAPPDATA\${COMPANY}"
  InstallDir "$LOCALAPPDATA\${COMPANY}\${APPNAME}-test"
!else
  RequestExecutionLevel admin
  !define REGROOT HKLM
  !define PARENTDIR "$PROGRAMFILES64\${COMPANY}"
  InstallDir "$PROGRAMFILES64\${COMPANY}\${APPNAME}"
!endif
InstallDirRegKey ${REGROOT} "Software\${COMPANY}\${APPNAME}" "InstallDir"

VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "${APPNAME}"
VIAddVersionKey "CompanyName" "${COMPANY}"
VIAddVersionKey "FileDescription" "${APPNAME} installer"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "${COMPANY}"

!include "MUI2.nsh"
!include "x64.nsh"
!include "FileFunc.nsh"

!define MUI_ICON "..\src\icon.ico"
!define MUI_UNICON "..\src\icon.ico"
!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TEXT "This installs ${APPNAME}, a GPU Mandelbrot viewer for 4K HDR and deep zooms.$\r$\n$\r$\nIt needs a 64-bit Windows PC with an NVIDIA GPU (driver with CUDA 13 support).$\r$\n$\r$\nClick Next to continue."
!define MUI_FINISHPAGE_RUN "$INSTDIR\${EXE}"
!define MUI_FINISHPAGE_RUN_TEXT "Launch ${APPNAME}"
!define MUI_FINISHPAGE_TEXT "${APPNAME} is installed.$\r$\n$\r$\nZoom video export uses ffmpeg. Install it with 'winget install Gyan.FFmpeg' or copy ffmpeg.exe next to ${EXE}. Screenshots go to Pictures\${APPNAME}, videos to Videos\${APPNAME}, presets to %APPDATA%\${COMPANY}\${APPNAME}."
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "${APPNAME} needs 64-bit Windows."
    Abort
  ${EndIf}
!ifndef TESTMODE
  SetShellVarContext all
!endif
FunctionEnd

Function un.onInit
!ifndef TESTMODE
  SetShellVarContext all
!endif
FunctionEnd

Section "${APPNAME} (required)" SEC_APP
  SectionIn RO
  SetOutPath "$INSTDIR"
  File "${STAGE}\${EXE}"
  File "${STAGE}\*.dll"
  File "${STAGE}\README.txt"
  WriteUninstaller "$INSTDIR\Uninstall.exe"

  WriteRegStr ${REGROOT} "Software\${COMPANY}\${APPNAME}" "InstallDir" "$INSTDIR"
  WriteRegStr ${REGROOT} "${UNINSTKEY}" "DisplayName" "${APPNAME}"
  WriteRegStr ${REGROOT} "${UNINSTKEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr ${REGROOT} "${UNINSTKEY}" "Publisher" "${COMPANY}"
  WriteRegStr ${REGROOT} "${UNINSTKEY}" "DisplayIcon" "$INSTDIR\${EXE}"
  WriteRegStr ${REGROOT} "${UNINSTKEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr ${REGROOT} "${UNINSTKEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr ${REGROOT} "${UNINSTKEY}" "QuietUninstallString" '"$INSTDIR\Uninstall.exe" /S'
  WriteRegDWORD ${REGROOT} "${UNINSTKEY}" "NoModify" 1
  WriteRegDWORD ${REGROOT} "${UNINSTKEY}" "NoRepair" 1
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD ${REGROOT} "${UNINSTKEY}" "EstimatedSize" "$0"
SectionEnd

Section "Start Menu shortcut" SEC_SM
  CreateDirectory "$SMPROGRAMS\${COMPANY}"
  CreateShortcut "$SMPROGRAMS\${COMPANY}\${APPNAME}.lnk" "$INSTDIR\${EXE}"
SectionEnd

Section /o "Desktop shortcut" SEC_DESK
  CreateShortcut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\${EXE}"
SectionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_APP} "The viewer and its runtime libraries."
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_SM} "Shortcut under Start > ${COMPANY}."
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_DESK} "Shortcut on the desktop."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

Section "Uninstall"
  Delete "$INSTDIR\${EXE}"
  Delete "$INSTDIR\*.dll"
  Delete "$INSTDIR\README.txt"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
  RMDir "${PARENTDIR}"
  Delete "$SMPROGRAMS\${COMPANY}\${APPNAME}.lnk"
  RMDir "$SMPROGRAMS\${COMPANY}"
  Delete "$DESKTOP\${APPNAME}.lnk"
  DeleteRegKey ${REGROOT} "${UNINSTKEY}"
  DeleteRegKey ${REGROOT} "Software\${COMPANY}\${APPNAME}"
  DeleteRegKey /ifempty ${REGROOT} "Software\${COMPANY}"
  ; User data in %APPDATA%\NMS\Mandelbloom (presets) is kept on purpose.
SectionEnd

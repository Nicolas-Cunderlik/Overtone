; Overtone installer (Inno Setup 6). Build it with scripts/build_installer.py, which
; passes AppVersion, PythonVersion and BuildDir (the wheels, pylock.toml and uv.exe).
;
; Setup copies those into {app}\install, then uses uv to install Python, create
; {app}\.venv and install the locked packages, fetches the catalog index into
; {app}\models, and runs `overtone --self-test`, which downloads the model weights
; into {app}\weights. Running setup again resumes or upgrades in place.
; The user's own files (login token, logs, caches) are in %LOCALAPPDATA%\Overtone
; and survive uninstalling.

#ifndef AppVersion
  #error Build with scripts/build_installer.py
#endif

#define AppName "Overtone"
#define GuiExe "{app}\.venv\Scripts\overtone-gui.exe"

[Setup]
AppId={{71C73F29-38C4-46BA-ADB4-F847C1CB7036}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Nicolas Cunderlik
AppPublisherURL=https://github.com/Nicolas-Cunderlik/Overtone
DefaultDirName={localappdata}\Programs\{#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
; Downloaded during setup: ~5 GB of packages, ~4 GB of model weights, the 100 MB index
ExtraDiskSpaceRequired=10737418240
InfoBeforeFile=..\THIRD_PARTY_NOTICES.md
OutputBaseFilename={#AppName}-setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName={#AppName}
UninstallDisplayIcon={#GuiExe}
CloseApplications=yes
SetupLogging=yes
; Off: the commands Setup runs inherit it (despite the help saying otherwise), and it blocks the
; junction uv makes for its Python (error 448). Setup isn't elevated and only writes to the user's
; own folders, so there's no privilege boundary for the mitigation to protect.
RedirectionGuard=no

[InstallDelete]
; Last version's wheels and lock file
Type: filesandordirs; Name: "{app}\install"

[Files]
Source: "{#BuildDir}\uv.exe"; DestDir: "{app}\install"
Source: "{#BuildDir}\pylock.toml"; DestDir: "{app}\install"
Source: "{#BuildDir}\wheels\*.whl"; DestDir: "{app}\install\wheels"
Source: "..\LICENSE"; DestDir: "{app}"
Source: "..\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{#GuiExe}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}"; Filename: "{#GuiExe}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{#GuiExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent; Check: SetupSucceeded

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\models"
Type: filesandordirs; Name: "{app}\weights"
Type: filesandordirs; Name: "{app}\uv-cache"
Type: filesandordirs; Name: "{app}\install"
Type: files; Name: "{app}\.env"

[Code]
var
  KeysPage: TInputQueryWizardPage;
  Succeeded: Boolean;
  FailedStep: String;

function AppPath(const Name: String): String;
begin
  Result := ExpandConstant('{app}\') + Name;
end;

function LogPath: String;
begin
  Result := AppPath('install\install.log');
end;

function SetupSucceeded: Boolean;
begin
  Result := Succeeded;
end;

procedure InitializeWizard;
begin
  KeysPage := CreateInputQueryPage(wpSelectTasks, 'API keys', 'Your Spotify and OpenAI keys',
    'Paste the Client ID and secret of your Spotify developer app (developer.spotify.com/dashboard, ' +
    'with the redirect URI below added to it; the Install section of the README on GitHub shows how). ' +
    'The OpenAI key is only needed for chord suggestions. The keys are saved in the .env file in the ' +
    'install folder, which you can edit later, and are only sent to Spotify and OpenAI.');
  KeysPage.Add('Spotify client ID:', False);
  KeysPage.Add('Spotify client secret:', True);
  KeysPage.Add('Spotify redirect URI:', False);
  KeysPage.Add('OpenAI API key (optional):', True);
  KeysPage.Values[2] := 'http://127.0.0.1:4000';
end;

// An upgrade keeps the keys already saved
function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := (PageID = KeysPage.ID) and FileExists(AppPath('.env'));
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = KeysPage.ID) and ((Trim(KeysPage.Values[0]) = '') or (Trim(KeysPage.Values[1]) = '')) then
  begin
    MsgBox('Enter your Spotify client ID and client secret.', mbError, MB_OK);
    Result := False;
  end;
end;

procedure WriteEnvFile;
var
  Lines: TArrayOfString;
begin
  if FileExists(AppPath('.env')) then
    Exit;
  SetArrayLength(Lines, 4);
  Lines[0] := 'SPOTIPY_CLIENT_ID=' + Trim(KeysPage.Values[0]);
  Lines[1] := 'SPOTIPY_CLIENT_SECRET=' + Trim(KeysPage.Values[1]);
  Lines[2] := 'SPOTIPY_REDIRECT_URI=' + Trim(KeysPage.Values[2]);
  Lines[3] := 'OPENAI_API_KEY=' + Trim(KeysPage.Values[3]);
  // Plain ASCII without a BOM: overtone_core.secrets reads it line by line
  SaveStringsToFile(AppPath('.env'), Lines, False);
end;

// Runs a command line through cmd, hidden, appending its output to the log
function RunStep(const Status, Command: String): Boolean;
var
  ResultCode: Integer;
begin
  WizardForm.StatusLabel.Caption := Status;
  WizardForm.FilenameLabel.Caption := '';
  SaveStringToFile(LogPath, #13#10 + '=== ' + Status + #13#10, True);
  Log('Running: ' + Command);
  Result := Exec(ExpandConstant('{cmd}'), '/S /C "' + Command + ' >> ' + AddQuotes(LogPath) + ' 2>&1"',
    AppPath(''), SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
  if not Result then
  begin
    FailedStep := Status;
    Log(Format('Failed with exit code %d', [ResultCode]));
  end;
end;

// The venv exists and runs the pinned Python (uv writes version_info to pyvenv.cfg)
function VenvMatches: Boolean;
var
  Lines: TArrayOfString;
  I: Integer;
begin
  Result := False;
  if LoadStringsFromFile(AppPath('.venv\pyvenv.cfg'), Lines) then
    for I := 0 to GetArrayLength(Lines) - 1 do
      if Trim(Lines[I]) = 'version_info = {#PythonVersion}' then
        Result := True;
end;

procedure InstallEnvironment;
var
  Env, Uv, Python: String;
begin
  // uv's Python and cache stay in the install folder; the user's uv config is ignored
  Env := 'set "UV_PYTHON_INSTALL_DIR=' + AppPath('python') + '"&& set "UV_CACHE_DIR=' + AppPath('uv-cache') +
    '"&& set "UV_NO_CONFIG=1"&& set "HF_HUB_DISABLE_PROGRESS_BARS=1"&& ';
  Uv := Env + AddQuotes(AppPath('install\uv.exe'));
  Python := AddQuotes(AppPath('.venv\Scripts\python.exe'));

  Succeeded := RunStep('Installing Python {#PythonVersion}...',
    Uv + ' python install {#PythonVersion} --no-bin --no-registry');
  if Succeeded and not VenvMatches then
    Succeeded := RunStep('Creating the Python environment...',
      Uv + ' venv --clear --managed-python --no-python-downloads --python {#PythonVersion} ' + AddQuotes(AppPath('.venv')));
  if Succeeded then
    Succeeded := RunStep('Downloading torch and the other packages (about 5 GB on a first install)...',
      Uv + ' pip sync --require-hashes --python ' + Python + ' ' + AddQuotes(AppPath('install\pylock.toml')));
  // --reinstall --no-cache: always these wheels. uv's cache (kept after a failed run) can hold an
  // older build with the same version, which it would otherwise install instead.
  if Succeeded then
    Succeeded := RunStep('Installing Overtone...',
      Uv + ' pip install --reinstall --no-cache --no-deps --no-index --find-links ' + AddQuotes(AppPath('install\wheels')) +
      ' --python ' + Python + ' overtone overtone-core');
  if Succeeded then
    Succeeded := RunStep('Downloading the song catalog index (about 100 MB)...',
      Env + Python + ' -m overtone_core.ml.model_release fetch');
  if Succeeded then
    Succeeded := RunStep('Downloading the audio models (about 4 GB on a first install) and testing Overtone...',
      Env + AddQuotes(AppPath('.venv\Scripts\overtone.exe')) + ' --self-test');
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep <> ssPostInstall then
    Exit;
  WriteEnvFile;
  WizardForm.ProgressGauge.Style := npbstMarquee;
  InstallEnvironment;
  WizardForm.ProgressGauge.Style := npbstNormal;
  if Succeeded then
    // The venv holds its own hard links or copies of the cached files, so they stay
    DelTree(AppPath('uv-cache'), True, True, True)
  else
    MsgBox('Setup stopped at: ' + FailedStep + #13#10#13#10 + 'Details are in ' + LogPath + '.' + #13#10#13#10 +
      'Check your internet connection and free disk space, then run the installer again: it continues where it stopped.',
      mbCriticalError, MB_OK);
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if (CurPageID = wpFinished) and not Succeeded then
    WizardForm.FinishedLabel.Caption := 'Overtone is not ready yet: setup stopped at "' + FailedStep + '". ' +
      'Details are in ' + LogPath + '.' + #13#10#13#10 + 'Run the installer again to finish.';
end;

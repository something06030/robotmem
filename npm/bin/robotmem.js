#!/usr/bin/env node
'use strict';

const { spawnSync } = require('child_process');
const os = require('os');

const args = process.argv.slice(2);
const isWin = os.platform() === 'win32';

// --- helpers ---

function run(cmd, cmdArgs, opts) {
  return spawnSync(cmd, cmdArgs, { stdio: 'inherit', shell: isWin, ...opts });
}

function hasCmd(cmd) {
  const r = spawnSync(isWin ? 'where' : 'which', [cmd], { stdio: 'pipe', shell: isWin });
  return r.status === 0;
}

function fail(msg) {
  process.stderr.write(msg + '\n');
  process.exit(1);
}

// --- main ---

// 1. Already installed? Forward directly.
for (const py of ['python3', 'python']) {
  if (!hasCmd(py)) continue;
  const check = spawnSync(py, ['-c', 'import robotmem'], { stdio: 'pipe', shell: isWin });
  if (check.status === 0) {
    const r = run(py, ['-m', 'robotmem', ...args]);
    process.exit(r.status || 0);
  }
}

// 2. Not installed. Try auto-install.
process.stderr.write('robotmem not found, attempting auto-install...\n');

if (hasCmd('pipx')) {
  process.stderr.write('→ pipx install robotmem\n');
  const install = run('pipx', ['install', 'robotmem']);
  if (install.status !== 0) {
    fail('❌ pipx install failed. Run manually: pipx install robotmem');
  }
} else if (hasCmd('pip3')) {
  process.stderr.write('→ pip3 install robotmem\n');
  const install = run('pip3', ['install', 'robotmem']);
  if (install.status !== 0) {
    fail('❌ pip3 install failed. Run manually: pip3 install robotmem');
  }
} else if (hasCmd('pip')) {
  process.stderr.write('→ pip install robotmem\n');
  const install = run('pip', ['install', 'robotmem']);
  if (install.status !== 0) {
    fail('❌ pip install failed. Run manually: pip install robotmem');
  }
} else {
  fail(
    '❌ No pipx or pip found. Install Python first:\n' +
    '   macOS:   brew install python\n' +
    '   Ubuntu:  sudo apt install python3-pip\n' +
    '   Windows: https://python.org/downloads\n\n' +
    '   Then run: pipx install robotmem'
  );
}

// 3. Run the command after installation.
for (const py of ['python3', 'python']) {
  if (!hasCmd(py)) continue;
  const check = spawnSync(py, ['-c', 'import robotmem'], { stdio: 'pipe', shell: isWin });
  if (check.status === 0) {
    const r = run(py, ['-m', 'robotmem', ...args]);
    process.exit(r.status || 0);
  }
}

fail(
  '❌ Installed but robotmem not in PATH.\n' +
  '   Try: pipx ensurepath && source ~/.bashrc\n' +
  '   Then re-run the command.'
);

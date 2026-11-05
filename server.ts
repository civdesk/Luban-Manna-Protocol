import express from 'express';
import { createServer as createViteServer } from 'vite';
import { exec, spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT) || 3000;

  app.use(express.json());

  // API to get current code of manna_rl_controller.py
  app.get('/api/code', (_req, res) => {
    try {
      const codePath = path.resolve(__dirname, 'manna_rl_controller.py');
      if (fs.existsSync(codePath)) {
        const content = fs.readFileSync(codePath, 'utf-8');
        return res.json({ success: true, code: content });
      }
      return res.status(404).json({ success: false, error: 'File not found' });
    } catch (err: any) {
      return res.status(500).json({ success: false, error: err.message });
    }
  });

  // API to trigger Python RL simulation and return execution log & metrics
  app.post('/api/run-simulation', async (_req, res) => {
    try {
      const pythonScript = path.resolve(__dirname, 'manna_rl_controller.py');
      const process = spawn('python3', [pythonScript], { cwd: __dirname });

      let stdout = '';
      let stderr = '';

      process.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      process.on('close', (code) => {
        const success = code === 0;

        // Parse key metrics if present
        let metrics: Record<string, any> = {};
        const vxMatch = stdout.match(/平均轴向推进速度.*?:\s*([0-9.-]+)\s*μm\/s/);
        const powerMatch = stdout.match(/平均机械做功功耗.*?:\s*([0-9.eE+-]+)\s*W\s*\(([0-9.-]+)\s*fW\)/);
        const shearMatch = stdout.match(/平均壁面剪切应力.*?:\s*([0-9.eE+-]+)\s*N\/m²/);
        const efficiencyMatch = stdout.match(/综合推进效率.*?:\s*([0-9.eE+-]+)/);
        const fWEfficiencyMatch = stdout.match(/工程标准化效率.*?:\s*([0-9.-]+)/);
        const shearStatusMatch = stdout.match(/剪切应力安全冗余.*?:\s*(\[[^\]]+\])/);

        if (vxMatch) metrics.vx = parseFloat(vxMatch[1]);
        if (powerMatch) {
          metrics.powerW = parseFloat(powerMatch[1]);
          metrics.powerFw = parseFloat(powerMatch[2]);
        }
        if (shearMatch) metrics.shearStress = parseFloat(shearMatch[1]);
        if (efficiencyMatch) metrics.propulsionEfficiency = parseFloat(efficiencyMatch[1]);
        if (fWEfficiencyMatch) metrics.fwEfficiency = parseFloat(fWEfficiencyMatch[1]);
        if (shearStatusMatch) metrics.shearStatus = shearStatusMatch[1];

        res.json({
          success,
          exitCode: code,
          stdout,
          stderr,
          metrics,
        });
      });
    } catch (err: any) {
      res.status(500).json({ success: false, error: err.message });
    }
  });

  // API to push to GitHub
  app.post('/api/github/push', async (req, res) => {
    const { token, repo = 'civdesk/Luban-Manna-Protocol', branch = 'main', message = 'feat: add Manna nanorobot RL controller with Stokesian physics & 1000-step test' } = req.body;

    if (!token) {
      return res.status(400).json({
        success: false,
        error: 'GitHub Personal Access Token (PAT) is required to authenticate push to private/user repositories.',
      });
    }

    try {
      // Execute git commands safely
      const remoteUrl = `https://${token}@github.com/${repo}.git`;

      // 1. Initialize or config git
      await new Promise((resolve, reject) => {
        exec('git init && git config user.name "civdesk" && git config user.email "annazhang.moisson@gmail.com"', { cwd: __dirname }, (err) => {
          if (err) reject(err);
          else resolve(true);
        });
      });

      // 2. Add and commit manna_rl_controller.py
      await new Promise((resolve, reject) => {
        exec(`git add manna_rl_controller.py && git commit -m "${message.replace(/"/g, '\\"')}" || true`, { cwd: __dirname }, (err) => {
          if (err) reject(err);
          else resolve(true);
        });
      });

      // 3. Get commit hash
      const commitHash: string = await new Promise((resolve) => {
        exec('git rev-parse HEAD', { cwd: __dirname }, (err, stdout) => {
          if (err) resolve('unknown');
          else resolve(stdout.trim());
        });
      });

      // 4. Rename branch to target branch
      await new Promise((resolve) => {
        exec(`git branch -M ${branch}`, { cwd: __dirname }, () => resolve(true));
      });

      // 5. Push to GitHub
      const pushResult: { stdout: string; stderr: string } = await new Promise((resolve, reject) => {
        exec(`git push --force ${remoteUrl} ${branch}`, { cwd: __dirname }, (err, stdout, stderr) => {
          if (err) {
            reject(new Error(stderr || err.message));
          } else {
            resolve({ stdout, stderr });
          }
        });
      });

      const commitUrl = `https://github.com/${repo}/commit/${commitHash}`;

      return res.json({
        success: true,
        repo,
        branch,
        commitHash,
        commitUrl,
        output: pushResult.stdout || pushResult.stderr || 'Pushed successfully',
      });
    } catch (err: any) {
      // Strip token from error message to protect secrets
      const safeError = (err.message || '').replace(new RegExp(token, 'g'), '***TOKEN***');
      return res.status(500).json({
        success: false,
        error: safeError,
      });
    }
  });

  // Check repo metadata via GitHub public API
  app.get('/api/github/check-repo', async (_req, res) => {
    try {
      const response = await fetch('https://api.github.com/repos/civdesk/Luban-Manna-Protocol', {
        headers: { 'User-Agent': 'Manna-App' }
      });
      if (response.ok) {
        const data = await response.json();
        return res.json({ exists: true, data });
      }
      return res.json({ exists: false, status: response.status });
    } catch (err: any) {
      return res.status(500).json({ error: err.message });
    }
  });

  // Vite development middleware
  const isProd = process.env.NODE_ENV === 'production';
  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();

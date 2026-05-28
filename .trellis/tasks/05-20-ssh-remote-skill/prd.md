# Create SSH Remote Connection Skill

## Goal

Create a Codex skill for SSH remote connection workflows. The skill should keep connection values under its own `scripts/` directory and provide deterministic scripts for connecting and updating credentials.

## What I already know

- The user wants a new skill for SSH remote connection.
- The user wants password, username, port, and IP stored under the skill's `scripts/` directory.
- Current `~/.ssh/config` has an `ascend` host with:
  - HostName: `106.75.235.239`
  - Port: `32222`
  - User: `root+vm-BD9SyUOwTzQlMW3Z`
- The current repository has project-local skills under `.agents/skills/`.

## Requirements

- Add a new project skill named `ssh-remote-connect`.
- Store SSH connection values under `.agents/skills/ssh-remote-connect/scripts/`.
- Do not print or expose the password in normal output.
- Provide a script to connect with the stored values.
- Provide a script to set or update the password locally.
- Keep the local credentials file out of git.

## Acceptance Criteria

- [ ] `SKILL.md` has clear trigger guidance and concise workflow.
- [ ] `scripts/connect.sh` can read the local credential file and run `ssh`.
- [ ] `scripts/set-password.sh` can update the password without echoing it.
- [ ] `scripts/connection.env.example` documents required fields.
- [ ] `scripts/connection.local.env` exists locally with current host/user/port defaults.
- [ ] Validation passes with the skill validator.

## Out of Scope

- No remote connection attempt is required.
- No password is requested or displayed in chat.
- No dependency installation is performed automatically.

## Technical Notes

- Use `sshpass` only when it already exists and a password is configured.
- Without `sshpass`, fall back to regular `ssh` so the terminal can prompt for password.

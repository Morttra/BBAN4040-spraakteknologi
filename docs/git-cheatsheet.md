# Git and GitHub cheatsheet

Quick reference for using Git and GitHub in this project, written for first-time users.

The everyday loop is **pull → edit → add → commit → push**. The rest of this page covers setup, working in parallel and fixing mistakes.

## How Git moves your work

```
  YOUR MAC                                                          GITHUB
  your folder ──git add──► staging ──git commit──► local commits ──git push──► origin/main
       ▲                                                                            │
       └──────────────────────────────── git pull ──────────────────────────────────┘
```

- **Repository (repo):** the project folder plus its full history.
- **Staging area:** the changes you have picked for the next commit.
- **Commit:** a saved snapshot with a message. Only commits can be pushed.
- **origin / main:** `origin` is the copy on GitHub, `main` is the main branch.
- **Branch:** a parallel line of work that can be merged back into `main`.
- **Git vs GitHub:** Git is the tool on your Mac; GitHub is the website that holds the shared copy.

Your partner has the same setup on their own Mac, and GitHub is the only place you meet. Nothing you do reaches your partner until you push, and nothing they do reaches you until you pull.

## One-time setup

1. **Add the partner.** The owner (Morttra) opens the repo on GitHub → Settings → Collaborators → Add people. The partner accepts the email invitation.
2. **Tell Git who you are** (once per Mac):

   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"   # the email on your GitHub account
   git config --global pull.rebase false               # `git pull` merges (avoids a confusing warning)
   git config --global core.editor "code --wait"       # Git opens VS Code instead of vim
   ```

   The last line needs the `code` command: in VS Code, press ⌘⇧P and run "Shell Command: Install 'code' command in PATH".
3. **Sign in to GitHub.** GitHub does not accept your password in the terminal. The easiest way is to accept when VS Code asks you to "Sign in with GitHub" on your first push. The terminal alternative is `brew install gh`, then `gh auth login` and `gh auth setup-git`.
4. **Get the repo** (partner only, the owner already has it). Go to the folder where you keep projects, then:

   ```bash
   git clone https://github.com/Morttra/BBAN4040-spraakteknologi.git
   cd BBAN4040-spraakteknologi
   ```

5. **Set up the project** (each person). Files that are not in Git must be created on each Mac:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   cp .env.example .env      # then put your own keys in .env
   ```

## The everyday loop

```bash
git pull                                              # 1. get your partner's latest work, ALWAYS first
# ... edit files ...
git status                                            # 2. see what changed
git add src/norrag/chunking.py                        # 3. stage files (`git add .` stages everything listed)
git commit -m "Add chunking with stable chunk ids"    # 4. save a snapshot
git push                                              # 5. share it on GitHub
```

- **Pull when you start, push when you stop.** Small, frequent commits cause fewer conflicts than one big commit at the end, and the course wants the history to show how the project developed.
- **Run `git status` before `git add .`** and check that nothing unexpected is listed (data, `.env`, large files).
- **Commit messages say what the commit does:** "Add BM25 retriever", "Fix hit@k when fewer than k results". Avoid "update" or "stuff".
- **Push rejected with "fetch first" or "non-fast-forward"?** Your partner pushed in the meantime. Run `git pull`, then `git push` again.

## Working on a branch

A branch lets one of you build something larger (a new retriever, the judge) without breaking `main` for the other. A simple rule for two people: small fixes go straight on `main`, and anything that takes more than one session goes on a branch.

```bash
git switch main && git pull           # start from the newest main
git switch -c bm25-retriever          # create a branch and move to it
# ... edit, add and commit as usual ...
git push -u origin bm25-retriever     # first push of the branch (after that, just `git push`)
```

Then on GitHub, open the repo → "Compare & pull request" → describe the change → "Create pull request". The other person looks through the changes, comments if needed and clicks "Merge pull request". Afterwards, both of you run:

```bash
git switch main
git pull
git branch -d bm25-retriever          # delete the merged branch locally
```

- `git branch` lists your branches; the current one is marked with `*`.
- `git switch <name>` moves between branches. Commit or stash your changes first.

## Checking what's going on

| Command | What it shows |
|---|---|
| `git status` | Changed, staged and new files, and which branch you are on |
| `git diff` | Line-by-line changes you have not staged yet |
| `git diff --staged` | What will go into the next commit |
| `git log --oneline` | The commit history, one line per commit |
| `git log --oneline --graph --all` | The history of all branches as a graph |
| `git show <hash>` | What one commit changed (the hash comes from `git log`, e.g. `a1b2c3d`) |
| `git branch -a` | Your branches and the ones on GitHub |
| `git remote -v` | Where `origin` points |

Press `q` to leave the `log` and `diff` views.

## Fixing a merge conflict

A conflict means you both changed the same lines and Git cannot choose. It is normal, not an error.

1. `git pull` reports `CONFLICT (content): Merge conflict in <file>`.
2. Open the file. Git has marked the clash:

   ```
   <<<<<<< HEAD
   your version of the lines
   =======
   your partner's version of the lines
   >>>>>>> 4e5f6a7
   ```

3. Keep the version you want and delete the three marker lines. VS Code shows buttons above the conflict: Accept Current, Accept Incoming, Accept Both.
4. Run `git add <file>`, then `git commit --no-edit` (keeps Git's suggested message), then `git push`.

- **Escape hatch:** `git merge --abort` puts everything back to how it was before the pull.
- **Notebook conflicts are hard to fix by hand,** because a `.ipynb` file is JSON. Avoid them by never editing the same notebook at the same time.

## Undoing things

Ordered from safe to risky.

| Situation | Command | Risk |
|---|---|---|
| Unstage a file but keep the changes | `git restore --staged <file>` | Safe |
| Put work aside to pull or switch branch | `git stash`, later `git stash pop` | Safe |
| Fix the last commit's message, or add a forgotten file, before pushing | `git add <file>`, then `git commit --amend` | Safe before push; never amend a pushed commit |
| Undo a commit that is already pushed | `git revert <hash>`, then `git push` | Safe: adds a new commit that reverses it |
| Stop tracking a file committed by mistake (it stays on disk) | `git rm --cached <file>`, add it to `.gitignore`, commit | Safe |
| Throw away your edits to one file | `git restore <file>` | Deletes those edits for good |
| Throw away all uncommitted work | `git reset --hard` | Deletes all uncommitted work for good |
| Overwrite the history on GitHub | `git push --force` | Don't: it can delete your partner's commits |

## Rules for this repo

- **Never commit** `.env` (API keys), `data/`, `results/raw/` or model files. `.gitignore` already blocks them, and `git status` before committing confirms they are not listed.
- **GitHub rejects files over 100 MB.** If `git status` shows a large file you did not expect, do not add it.
- **If a key ends up in a commit, revoke it at once** (OpenAI: API keys page; Hugging Face: Settings → Access Tokens) and create a new one. Deleting the file afterwards does not help, because the key stays in the history.
- **Agree on who edits which notebook.** Pull before you open one and push soon after you close it.

## The same in VS Code

The Source Control view is the branch icon in the left bar (⌃⇧G).

| Command | In VS Code |
|---|---|
| `git status`, `git diff` | Source Control view; click a file to see its changes |
| `git add <file>` | `+` next to the file (Stage Changes) |
| `git commit -m "..."` | Type the message in the box and click Commit |
| `git pull`, `git push` | "Sync Changes" button (pulls, then pushes), or the `…` menu → Pull / Push |
| `git switch`, `git switch -c` | Branch name in the bottom-left status bar → pick or create a branch |
| Merge conflicts | Open the file → Accept Current / Incoming / Both, or "Resolve in Merge Editor" |

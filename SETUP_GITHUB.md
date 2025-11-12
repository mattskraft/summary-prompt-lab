# Setting up GitHub Repository for Summary Prompt Lab

## Pre-flight Checklist

### ✅ Files to Commit
- [x] All source code (`apps/`, `src/`)
- [x] Configuration files (`pyproject.toml`, `requirements.txt`)
- [x] Documentation (`README.md`)
- [x] `.gitignore` (already configured)
- [x] `config/.env.example` (template, safe to commit)
- [x] `data/processed/.gitkeep` (keeps directory structure)

### ⚠️ Files to EXCLUDE (already in .gitignore)
- `config/.env` - Contains secrets (API keys)
- `__pycache__/` - Python cache
- `*.egg-info/` - Build artifacts
- `data/processed/*.json` - Large data files (symlinks)
- `.streamlit/` - Streamlit config

### 🔍 Before Committing

1. **Check for sensitive data:**
   ```bash
   # Make sure .env is not tracked
   git status
   ```

2. **Verify .gitignore is working:**
   ```bash
   git check-ignore config/.env
   # Should output: config/.env
   ```

3. **Check file sizes:**
   ```bash
   # Large files should be excluded
   find . -type f -size +1M
   ```

## Step-by-Step: Create GitHub Repository

### 1. Initialize Git Repository

```bash
cd /home/matthias/Kiso/code/projects/summary-prompt-lab

# Initialize git
git init

# Check what will be committed
git status
```

### 2. Create Initial Commit

```bash
# Add all files (respects .gitignore)
git add .

# Verify what's staged
git status

# Create initial commit
git commit -m "Initial commit: Summary Prompt Lab

- Streamlit app for exploring summary prompt generation
- Kiso input processing utilities
- Configuration management
- Documentation and setup files"
```

### 3. Create Repository on GitHub

**Option A: Using GitHub CLI (if installed)**
```bash
gh repo create summary-prompt-lab \
  --public \
  --description "Streamlit app for exploring and testing summary prompt generation for Kiso Mind" \
  --source=. \
  --remote=origin \
  --push
```

**Option B: Using GitHub Web Interface**
1. Go to https://github.com/new
2. Repository name: `summary-prompt-lab`
3. Description: "Streamlit app for exploring and testing summary prompt generation for Kiso Mind"
4. Choose visibility (public/private)
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### 4. Connect Local Repository to GitHub

```bash
# Add remote (replace USERNAME with your GitHub username)
git remote add origin https://github.com/USERNAME/summary-prompt-lab.git

# Or using SSH (if you have SSH keys set up)
# git remote add origin git@github.com:USERNAME/summary-prompt-lab.git

# Verify remote
git remote -v
```

### 5. Push to GitHub

```bash
# Push main branch
git branch -M main
git push -u origin main
```

## Post-Setup Considerations

### Repository Settings on GitHub

1. **Add topics/tags:**
   - `streamlit`
   - `python`
   - `nlp`
   - `prompt-engineering`
   - `kiso`

2. **Add repository description:**
   - "Streamlit app for exploring and testing summary prompt generation for Kiso Mind"

3. **Consider adding:**
   - License file (MIT, Apache 2.0, etc.)
   - Contributing guidelines (if open source)
   - Issue templates
   - GitHub Actions for CI/CD (optional)

### Security Best Practices

1. **Never commit:**
   - API keys
   - Personal data paths
   - `.env` files

2. **Use GitHub Secrets for CI/CD:**
   - If you add GitHub Actions later, use repository secrets

3. **Review .gitignore regularly:**
   - Ensure new sensitive files are covered

### Optional: Add License

```bash
# Example: Create MIT License
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted...
EOF

git add LICENSE
git commit -m "Add MIT License"
git push
```

## Troubleshooting

### If .env was accidentally committed:
```bash
# Remove from git (but keep local file)
git rm --cached config/.env
git commit -m "Remove .env from repository"
git push

# If already pushed, consider rotating API keys
```

### If large files were committed:
```bash
# Use git-filter-repo or BFG Repo-Cleaner
# Or contact GitHub support if repository is new
```

## Next Steps

1. ✅ Repository created and pushed
2. 📝 Update README with any additional setup instructions
3. 🔒 Verify all secrets are excluded
4. 🏷️ Add topics and description on GitHub
5. 📋 Consider adding a LICENSE file
6. 🤖 Set up GitHub Actions (optional, for CI/CD)


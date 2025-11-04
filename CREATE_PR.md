# How to Create Pull Request to mycpuorg/helion

## Quick Link (Easiest Method)

**Click here to create the PR:**

👉 **https://github.com/mycpuorg/helion/compare/main...claude/openevolve-autotuner-helion-011CUoUYodYsMMzqcBCnGbKR** 👈

Then:
1. Click the green "Create pull request" button
2. Copy the content from `PR_DESCRIPTION.md` into the description field
3. Click "Create pull request"

## Method 1: Web Interface (Recommended)

### Step 1: Visit the PR creation URL
```
https://github.com/mycpuorg/helion/pull/new/claude/openevolve-autotuner-helion-011CUoUYodYsMMzqcBCnGbKR
```

### Step 2: Set PR details
- **Base branch**: `main` (should be selected automatically)
- **Compare branch**: `claude/openevolve-autotuner-helion-011CUoUYodYsMMzqcBCnGbKR` (already selected)
- **Title**: `Add OpenEvolve-based Autotuner for Helion GPU Kernels`

### Step 3: Copy PR description
```bash
# On macOS
cat PR_DESCRIPTION.md | pbcopy

# On Linux
cat PR_DESCRIPTION.md | xclip -selection clipboard

# Or just open the file
cat PR_DESCRIPTION.md
```

### Step 4: Paste into GitHub and create

## Method 2: GitHub CLI (If Available)

```bash
gh pr create \
  --base main \
  --head claude/openevolve-autotuner-helion-011CUoUYodYsMMzqcBCnGbKR \
  --title "Add OpenEvolve-based Autotuner for Helion GPU Kernels" \
  --body-file PR_DESCRIPTION.md \
  --repo mycpuorg/helion
```

## Method 3: Using API (Advanced)

```bash
# Read the PR description
PR_BODY=$(cat PR_DESCRIPTION.md)

# Create PR via GitHub API
curl -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  https://api.github.com/repos/mycpuorg/helion/pulls \
  -d "{
    \"title\": \"Add OpenEvolve-based Autotuner for Helion GPU Kernels\",
    \"body\": $(jq -Rs . < PR_DESCRIPTION.md),
    \"head\": \"claude/openevolve-autotuner-helion-011CUoUYodYsMMzqcBCnGbKR\",
    \"base\": \"main\"
  }"
```

## PR Details Summary

- **Repository**: mycpuorg/helion
- **Base Branch**: main
- **Feature Branch**: claude/openevolve-autotuner-helion-011CUoUYodYsMMzqcBCnGbKR
- **Title**: Add OpenEvolve-based Autotuner for Helion GPU Kernels

### Files Changed
- 7 new files
- 2,500+ lines of code and documentation

### Key Features
- ✅ OpenEvolveTuner implementation
- ✅ B200-specific optimizations
- ✅ Comprehensive testing infrastructure
- ✅ Documentation and examples
- ✅ Backward compatible

## Verification

After creating the PR, verify:
1. Base branch is set to `main`
2. All 7 files are included in the PR
3. PR description is complete
4. CI checks pass (if configured)

## Need Help?

If you encounter issues:
1. Check that the branch is pushed: `git branch -r | grep claude/openevolve`
2. Verify remote: `git remote -v` (should show mycpuorg/helion)
3. Ensure you have push access to mycpuorg/helion

---

**Ready to create?** Use the quick link at the top! 🚀

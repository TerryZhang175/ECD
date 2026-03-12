## Skills
A skill is a set of local instructions to follow that is stored in a `SKILL.md` file.

### Available skills
- ecd-ui-debug: Debug and modify the ECD UI served from `/ui`, including `ui/index.html`, `ui/app.js`, `ui/styles.css`, and FastAPI static serving. Use when changing frontend behavior, investigating UI/backend mismatches, or when a fix seems correct in code but not visible in the browser. This skill includes cache-busting checks and browser-side verification. (file: /Users/terry/Codehub/ECD/.codex/skills/ecd-ui-debug/SKILL.md)
- ecd-response-style: Repository-specific response style for ECD work. Use when replying about this project so answers stay focused on the code/task and do not append generic product or workflow promotion unless the user explicitly asks for alternatives or a higher-priority instruction requires it. (file: /Users/terry/Codehub/ECD/.codex/skills/ecd-response-style/SKILL.md)

### How to use skills
- If the task clearly involves the ECD frontend, use `ecd-ui-debug`.
- Use `ecd-response-style` for normal user-facing replies in this repository.
- Read only the `SKILL.md` body unless more files are explicitly referenced from it.

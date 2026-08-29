# Resume Fit AI product workflow

Resume Fit AI is a resume-to-job-description matching and optimization application. A user pastes or uploads an existing resume, chooses a job-seeking identity and target role, and supplies the target job description (JD). The service returns a matching score, strengths, gaps, missing keywords, rewrite suggestions, project decisions, interview preparation, and an editable resume draft.

## Authoritative user inputs

The selected job-seeking identity is authoritative. A graduate profile and an experienced-hire profile follow different scoring priorities and resume section ordering. Work years are validated against the identity: an entry-level candidate claiming many years of employment, or an experienced hire claiming no work history, is stopped before analysis and asked to correct the conflict.

The career selector is a two-level tree delivered and validated by the server. The backend rejects a forged or mismatched parent-child career combination instead of trusting arbitrary browser values.

## Fast diagnosis and full generation

Fast diagnosis is a low-latency preview. The model returns the score, all problem titles, two detailed problems, two representative rewrites, one project diagnosis, and a small amount of interview preparation. The server renders the original resume layout immediately, so the preview does not wait for a complete AI rewrite.

Full generation rewrites the complete resume for the target role. It unlocks all problem explanations, the complete project plan, interview questions, a cover letter, direct resume editing, and PDF, DOCX, TXT, and Markdown exports. The two modes use different output budgets so latency and model cost match the product tier.

## Project decisions and factual boundaries

Each existing project receives a keep, strengthen, replace, or remove decision. When the resume and target role are far apart, the report proposes a new project with features, technology choices, and acceptance evidence. A proposed skill or project is a learning recommendation, not a claim that the candidate already has that experience.

The system must not invent employment, achievements, metrics, or technologies. Requirements found only in the JD belong in a gap or learning plan unless the original resume supports them.

## Input and export surface

The application accepts pasted text plus PDF, DOCX, and TXT resume uploads. Resume input is capped at 6,000 characters and JD input at 3,000 characters. Generated drafts preserve useful original sections such as grouped skills, campus activities, work history, and projects. The browser provides an A4 preview with classic and modern templates, theme color, font size, and line spacing controls.
